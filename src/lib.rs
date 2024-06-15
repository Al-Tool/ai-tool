#![feature(cfg_match)]
#![feature(str_from_raw_parts)]
#![feature(let_chains)]

use bincode::{deserialize_from, serialize_into};
use pyo3::{intern, prelude::*, types::PyType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use std::{
    collections::{HashMap, VecDeque}, fs::File, ops::{Deref, DerefMut}, ptr, mem::size_of
};

cfg_match! {
    cfg(test) => {
        use derive_debug::Dbg;
    }
}

#[repr(u8)]
#[pyclass]
#[derive(Clone)]
enum DatasetMode {
    Lines { length: usize },
    QA { sep: String },
    CSV { sep: String, slots: usize }
}

#[pyclass]
struct TextDataset {
    modes: DatasetMode,
    length: usize,
    content: MemoryWriter,
    torch: Py<PyModule>
}

#[pyclass]
#[derive(Clone)]
struct Tokenizer {
    vocabs: VocabTree,
}

struct MemoryReader {
    #[cfg(windows)]
    h_file: winapi::um::winnt::HANDLE,
    #[cfg(unix)]
    h_file: *mut libc::c_void,
    size: usize,
}

struct MemoryWriter {
    #[cfg(windows)]
    handle: winapi::um::winnt::HANDLE,
    #[cfg(unix)]
    handle: *mut libc::c_void,
    size: usize
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TokenCounter {
    curr: usize,
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(Dbg))]
struct VocabNode {
    token: usize,
    nexts: HashMap<char, VocabNode>,
}

#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(Dbg))]
struct VocabTree {
    trees: HashMap<char, VocabNode>,
    record: Vec<String>,
    counter: TokenCounter,
}

#[pyclass]
struct VocabTreeIter {
    stack: VecDeque<(VocabNode, String)>,
}

#[pymethods]
impl DatasetMode {
    #[classmethod]
    fn lines(_: &Bound<'_, PyType>, length: usize) -> Self {
        Self::Lines { length }
    }

    #[classmethod]
    fn qa(_: &Bound<'_, PyType>, sep: String) -> Self {
        Self::QA { sep }
    }
}

#[pymethods]
impl TextDataset {
    #[new]
    #[pyo3(text_signature="(file_path, tokenizer, modes=DatasetMode.lines(1), length=32)")]
    fn new(file_path: &str, tokenizer: &Tokenizer, modes: &DatasetMode, length: usize) -> PyResult<Self> {
        let reader = MemoryReader::new(file_path);
        let mut writer = MemoryWriter::new(tokenizer._tokens_len(&reader));
        tokenizer._encode_to(&reader, &mut writer)?;
        writer.lock();
        Ok(Self {
            modes: modes.clone(),
            length,
            content: writer,
            torch: Python::with_gil(|py| -> PyResult<Py<PyModule>> {
                let module = py.import_bound("torch")?;
                let unbound = module.unbind();
                Ok(unbound)
            })?
        })
    }

    fn __len__(slf: PyRef<'_, Self>) -> usize {
        slf.content.len() - slf.length + 1
    }

    fn __getitem__(slf: PyRef<'_, Self>, index: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let tokens = &slf.content[index..index + slf.length - 1];
        let mut x = tokens.to_vec();
        let mut y = tokens.to_vec();
        x.insert(0, 0);
        y.push(1);

        let py = slf.py();
        let tensor = slf.torch.getattr(py, intern!(py, "tensor"))?;
        let x = tensor.call1(py, (x,))?;
        let y = tensor.call1(py, (y,))?;
        Ok((x, y))
    }
}

const SPECIAL: [&str; 5] = ["[STT]", "[END]", "[UNK]", "[PAD]", "[SEP]"];

#[pymethods]
impl Tokenizer {
    #[new]
    fn new() -> Self {
        Self {
            vocabs: VocabTree::new(),
        }
    }

    #[pyo3(signature=(readable, times=10, min_frequency=0.0001, file=true))]
    fn train(
        mut slf: PyRefMut<'_, Self>,
        readable: &str,
        times: usize,
        min_frequency: f64,
        file: bool,
    ) -> PyResult<()> {
        if file {
            slf.train_from_file(readable, times, min_frequency)
        } else {
            slf.train_from_string(readable, times, min_frequency)
        }
    }

    #[inline]
    fn add_bases(mut slf: PyRefMut<'_, Self>, bases: &str) {
        for base in bases.chars() {
            slf.vocabs.add_line(&base.to_string());
        }
    }

    #[inline]
    fn add_token(mut slf: PyRefMut<'_, Self>, token: &str) {
        slf.vocabs.add_line(token);
    }

    fn encode(slf: PyRef<'_, Self>, content: &str) -> PyResult<Vec<usize>> {
        slf._encode(content)
    }

    fn decode(slf: PyRef<'_, Self>, tokens: Vec<usize>) -> PyResult<String> {
        slf._decode(tokens)
    }

    fn load(mut slf: PyRefMut<'_, Self>, file_name: &str) -> PyResult<()> {
        println!("loading vocabs from '{file_name}'...");
        slf.vocabs = deserialize_from(File::open(file_name)?).unwrap();
        println!("loaded success.");
        Ok(())
    }

    fn save(slf: PyRef<'_, Self>, file_name: &str) -> PyResult<()> {
        println!("saving vocabs to '{file_name}'...");
        serialize_into(File::create(file_name)?, &slf.vocabs).unwrap();
        println!("saved success.");
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> VocabTreeIter {
        slf.vocabs.clone().into_iter()
    }
}

#[derive(Clone)]
struct WindowStringIter<'a> {
    target: &'a str,
    start: usize,
    end: usize,
    index_buffer: VecDeque<usize>,
}

fn into_window<'life>(target: &'life str, length: usize) -> WindowStringIter<'life> {
    let mut slf = WindowStringIter {
        target,
        start: 0,
        end: 0,
        index_buffer: VecDeque::new(),
    };

    for (idx, c) in target.char_indices().take(length) {
        slf.index_buffer.push_back(idx);
        slf.end = idx + c.len_utf8();
    }

    slf
}

impl Tokenizer {
    fn _tokens_len(&self, content: &str) -> usize {
        let mut chars = content.chars();
        let mut result = 0usize;
        let mut curr_map = &self.vocabs.trees;
        let mut curr_node: Option<&VocabNode> = None;
        while let Some(mut c) = chars.next() {
            while !curr_map.contains_key(&c) {
                if std::ptr::eq(curr_map, &self.vocabs.trees) {
                    if let Some(nc) = chars.next() {
                        c = nc;
                    } else {
                        break;
                    }

                    result += 1;
                } else {
                    curr_map = &self.vocabs.trees;
                    result += 1;
                };
            }

            let node = curr_map.get(&c).unwrap();
            curr_node = Some(node);
            curr_map = &node.nexts;
        }

        if curr_node.is_some() {
            result += 1;
        }

        result
    }

    fn _encode_to(&self, content: &str, slice: &mut [usize]) -> PyResult<()> {
        let mut chars = content.chars();
        let mut idx = 0;
        let mut curr_map = &self.vocabs.trees;
        let mut curr_node: Option<&VocabNode> = None;
        while let Some(mut c) = chars.next() {
            while !curr_map.contains_key(&c) {
                if std::ptr::eq(curr_map, &self.vocabs.trees) {
                    slice[idx] = 2;
                    idx += 1;
                    if let Some(nc) = chars.next() {
                        c = nc;
                    } else {
                        break;
                    }
                } else {
                    slice[idx] = curr_node.take().unwrap().token;
                    idx += 1;
                    curr_map = &self.vocabs.trees;
                }
            }

            let node = curr_map.get(&c).unwrap();
            curr_node = Some(node);
            curr_map = &node.nexts;
        }

        if let Some(node) = curr_node {
            slice[idx] = node.token;
        }

        Ok(())
    }

    fn _encode(&self, content: &str) -> PyResult<Vec<usize>> {
        let mut chars = content.chars();
        let mut result = vec![0];
        let mut curr_map = &self.vocabs.trees;
        let mut curr_node: Option<&VocabNode> = None;
        while let Some(mut c) = chars.next() {
            while !curr_map.contains_key(&c) {
                if std::ptr::eq(curr_map, &self.vocabs.trees) {
                    result.push(2);
                    if let Some(nc) = chars.next() {
                        c = nc;
                    } else {
                        break;
                    }
                } else {
                    result.push(curr_node.take().unwrap().token);
                    curr_map = &self.vocabs.trees;
                }
            }

            let node = curr_map.get(&c).unwrap();
            curr_node = Some(node);
            curr_map = &node.nexts;
        }

        if let Some(node) = curr_node {
            result.push(node.token);
        }

        result.push(1);
        Ok(result)
    }

    fn _decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        Ok(tokens
            .into_iter()
            .filter(|token| *token > 4)
            .map(|token| self.vocabs.record[token].clone())
            .collect::<Vec<String>>()
            .concat())
    }

    fn train_from_file(
        &mut self,
        file_path: &str,
        times: usize,
        min_frequency: f64,
    ) -> PyResult<()> {
        self.train_from_string(&MemoryReader::new(file_path), times, min_frequency)
    }

    fn train_from_string(
        &mut self,
        content: &str,
        times: usize,
        min_frequency: f64,
    ) -> PyResult<()> {
        let chars = content.chars();
        for chr in chars.clone() {
            if !self.vocabs.find_root(chr) {
                self.vocabs.add_line(&chr.to_string());
            }
        }

        println!("prepare base tokens completed.");
        let per = (1..=times)
            .map(|i| 1.0 / (content.len() - i) as f64)
            .collect::<Vec<_>>();

        let frequencies = into_window(content, times + 1)
            .par_bridge()
            .fold(
                || vec![HashMap::new(); times],
                |mut p, slice| {
                    for (i, idx) in (2..=p.len()).zip(
                        slice
                            .char_indices()
                            .skip(2)
                            .take(p.len() - 1)
                            .map(|(idx, _)| idx),
                    ) {
                        if i >= slice.len() {
                            break;
                        }

                        let curr = per[i - 2];
                        let sub_slice = &slice[..idx];
                        if sub_slice.chars().skip(1).any(|c| !c.is_alphanumeric()) {
                            continue;
                        }

                        if let Some(r) = p[i - 2].get_mut(sub_slice) {
                            *r += curr;
                        } else {
                            p[i - 2].insert(sub_slice, curr);
                        }
                    }

                    p
                },
            )
            .reduce(
                || vec![HashMap::new(); times],
                |mut p, c| {
                    for (idx, other) in c.iter().enumerate() {
                        for (k, &v) in other {
                            if let Some(r) = p[idx].get_mut(k) {
                                *r += v;
                            } else {
                                p[idx].insert(k, v);
                            }
                        }
                    }

                    p
                },
            );

        for frequency in frequencies {
            for token in frequency
                .into_iter()
                .filter(|(_, v)| *v >= min_frequency)
                .map(|(k, _)| k)
            {
                self.vocabs.add_line(token);
            }
        }

        Ok(())
    }
}

unsafe impl Send for MemoryReader {}
unsafe impl Send for MemoryWriter {}
impl MemoryReader {
    #[cfg(windows)]
    fn new(file_path: &str) -> Self {
        use std::os::windows::io::AsRawHandle;
        use winapi::um::{
            memoryapi::{CreateFileMappingW, MapViewOfFile},
            winnt::{FILE_SHARE_READ, HANDLE, PAGE_READONLY},
        };

        let file = File::open(file_path).unwrap();
        let handle = file.as_raw_handle() as HANDLE;
        let size = file.metadata().unwrap().len() as usize;
        let mapping = unsafe {
            CreateFileMappingW(handle, ptr::null_mut(), PAGE_READONLY, 0, 0, ptr::null())
        };

        if mapping.is_null() {
            panic!("Unable to create FileMapping");
        }

        let h_file = unsafe { MapViewOfFile(mapping, FILE_SHARE_READ, 0, 0, size) };

        if h_file.is_null() {
            panic!("Unable to create MapView");
        }

        Self { h_file, size }
    }

    #[cfg(unix)]
    fn new(file_path: &str) -> Self {
        use libc::{mmap, MAP_FAILED, MAP_SHARED, PROT_READ};
        use std::os::fd::AsRawFd;

        let file = File::open(file_path).unwrap();
        let size = file.metadata().unwrap().len() as usize;
        let map = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                PROT_READ,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if map == MAP_FAILED {
            panic!("mmap create failed");
        }

        Self { h_file: map, size }
    }
}

impl MemoryWriter {
    #[cfg(windows)]
    fn new(size: usize) -> Self {
        use winapi::um::{memoryapi::VirtualAlloc, winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE}};

        let addr = unsafe { VirtualAlloc(
            ptr::null_mut(),
            size * size_of::<usize>(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE
        ) };

        if addr.is_null() {
            panic!("Failed to allocate memory");
        }

        Self {
            handle: addr,
            size
        }
    }

    #[cfg(unix)]
    fn new(size: usize) -> Self {
        use libc::{mmap, MAP_FAILED, PROT_WRITE, PROT_READ, MAP_ANON, MAP_PRIVATE};

        let addr = unsafe { mmap(
            ptr::null_mut(),
            size * size_of::<usize>(),
            PROT_READ | PROT_WRITE,
            MAP_ANON | MAP_PRIVATE,
            -1,
            0
        ) };

        if addr == MAP_FAILED {
            panic!("Failed to allocate memory");
        }

        Self {
            handle: addr,
            size
        }
    }

    #[cfg(windows)]
    fn lock(&mut self) {
        use winapi::um::{memoryapi::VirtualProtect, winnt::PAGE_READONLY};

        let mut old = 0;
        let succ = unsafe { VirtualProtect(
            self.handle,
            self.size * size_of::<usize>(),
            PAGE_READONLY,
            &mut old
        ) };
        if succ == 0 {
            panic!("Failed to lock memory to read");
        }
    }

    #[cfg(unix)]
    fn lock(&mut self) {
        use libc::{mprotect, PROT_READ};

        let succ = unsafe {
            mprotect(self.handle, self.size * size_of::<usize>(), PROT_READ)
        };

        if succ != 0 {
            panic!("Failed to lock memory to read");
        }
    }
}

impl Deref for MemoryReader {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        unsafe { std::str::from_raw_parts(self.h_file as *const _, self.size) }
    }
}

impl Deref for MemoryWriter {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.handle as *const _, self.size) }
    }
}

impl DerefMut for MemoryWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.handle as *mut _, self.size) }
    }
}

impl Drop for MemoryReader {
    #[cfg(windows)]
    fn drop(&mut self) {
        use winapi::um::memoryapi::UnmapViewOfFile;

        let succ = unsafe { UnmapViewOfFile(self.h_file) };
        if succ == 0 {
            panic!("Release mempry failed");
        }
    }

    #[cfg(unix)]
    fn drop(&mut self) {
        use libc::munmap;

        if unsafe { munmap(self.h_file, self.size) } != 0 {
            panic!("munmap release failed");
        }
    }
}

impl Drop for MemoryWriter {
    #[cfg(windows)]
    fn drop(&mut self) {
        use winapi::um::{memoryapi::VirtualFree, winnt::MEM_RELEASE};

        let succ = unsafe { VirtualFree(self.handle, 0, MEM_RELEASE) };
        if succ == 0 {
            panic!("Release mempry failed");
        }
    }

    #[cfg(unix)]
    fn drop(&mut self) {
        use libc::munmap;

        if unsafe { munmap(self.handle, self.size * size_of::<usize>()) } != 0 {
            panic!("munmap release failed");
        }
    }
}

#[pymethods]
impl VocabTreeIter {
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(String, usize)> {
        slf.next()
    }
}

impl TokenCounter {
    fn new(start: usize) -> Self {
        Self { curr: start }
    }

    fn next(&mut self) -> usize {
        let r = self.curr;
        self.curr += 1;
        r
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self { curr: 5 }
    }
}

impl VocabNode {
    fn new(counter: &mut TokenCounter) -> Self {
        let token = counter.next();
        Self {
            token,
            nexts: HashMap::new(),
        }
    }
}

impl VocabTree {
    fn new() -> Self {
        Self {
            trees: HashMap::new(),
            record: SPECIAL.map(ToString::to_string).to_vec(),
            counter: TokenCounter::new(5),
        }
    }

    #[inline]
    fn add_line(&mut self, line: &str) -> () {
        let mut nodes = &mut self.trees;
        let mut record_token = String::new();
        for c in line.chars() {
            record_token.push(c);
            if !nodes.contains_key(&c) {
                self.record.push(record_token.clone());
                nodes.insert(c, VocabNode::new(&mut self.counter));
            }

            nodes = &mut nodes.get_mut(&c).unwrap().nexts;
        }
    }

    #[inline]
    fn find_root(&self, base: char) -> bool {
        self.trees.contains_key(&base)
    }
}

impl IntoIterator for VocabTree {
    type Item = (String, usize);
    type IntoIter = VocabTreeIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let mut stack = VecDeque::new();
        for (k, root) in self.trees.iter() {
            stack.push_back((root.clone(), k.to_string()));
        }

        VocabTreeIter { stack }
    }
}

impl Iterator for VocabTreeIter {
    type Item = (String, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path)) = self.stack.pop_front() {
            let entries: Vec<(char, VocabNode)> = node.nexts.clone().into_iter().collect();
            for (k, child) in entries.into_iter().rev() {
                self.stack
                    .push_front((child, [path.clone(), k.to_string()].concat()));
            }

            if node.nexts.is_empty() {
                return Some((path, node.token));
            }
        }

        None
    }
}

impl<'life> Iterator for WindowStringIter<'life> {
    type Item = &'life str;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = &self.target[self.end..];
        let result = &self.target[self.start..self.end];
        if self.start == self.end {
            return None;
        }

        self.start = self.index_buffer.pop_front()?;
        if !curr.is_empty() {
            self.index_buffer.push_back(self.end);
            self.end += curr.chars().next()?.len_utf8();
        }

        Some(result)
    }
}

#[pymodule]
fn ai_tool(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatasetMode>()?;
    m.add_class::<TextDataset>()?;
    m.add_class::<Tokenizer>()?;
    m.add_class::<VocabTreeIter>()?;
    Ok(())
}

#[test]
fn test_vocab_tree() {
    let mut tree = VocabTree::new();
    tree.add_line("你好");
    tree.add_line("cls");
    tree.add_line("abc");
    tree.add_line("cd");
    tree.add_line("你是誰");

    for token in tree.into_iter() {
        println!("{token:?}");
    }
}

#[test]
fn test_chi_speed() {
    use num_cpus::get;
    use rayon::ThreadPoolBuilder;
    use std::time::Instant;

    fn process_line(line: &str) -> () {
        // 模拟处理，每个字符的 Unicode 值之和
        println!("{}", line.chars().map(|c| c.len_utf8()).sum::<usize>())
    }

    ThreadPoolBuilder::new()
        .num_threads(get())
        .build_global()
        .unwrap();
    let binding = MemoryReader::new("input.txt");
    binding
        .lines()
        .par_bridge()
        .map(process_line)
        .for_each(|_| {});
    let start = Instant::now();
    println!("{:?}", start.elapsed());
}

#[test]
fn test_tokenizer() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        let tokenizer = Tokenizer::new().into_py(py).into_bound(py);
        let mode = DatasetMode::Lines { length: 1 }.into_py(py).into_bound(py);
        tokenizer.getattr("load")?.call1(("mix.bin",))?;

        let dataset = TextDataset::new("input.txt", &tokenizer.extract()?, &mode.extract()?, 32)?.into_py(py).into_bound(py);

        Ok(())
    }).unwrap();
}

#[test]
fn test_window_slice() {
    let binding = MemoryReader::new("input.txt");
    into_window(&binding, 5)
        .take(20)
        .for_each(|slice| println!("{slice:?}"));
}
