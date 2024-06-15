# ai_tool

some useful ai-tools in this package for my project

## example

```python
from ai_tool import Tokenizer

tokenizer = Tokenizer()
tokenizer.add_token('token')
tokenizer.train('raw text here', file=False)
tokenizer.train('text_file.txt', times=10, min_frequency=0.001, alpha=0.0005)
tokenizer.save('vocab.bin')

new_tokenizer = Tokenizer()
new_tokenizer.load('vocab.bin')

print([token for token in new_tokenizer]) # longest tokens pair
```
