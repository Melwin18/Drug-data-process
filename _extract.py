src = open(r'C:\Users\asus\.gemini\antigravity-ide\brain\340dec3f-9c81-4e4f-a20a-c5966f69f32a\scratch\gen_corrected_pipeline.py', encoding='utf-8').read()
# Find the triple-quoted code block
marker = 'code = r'
idx = src.find(marker)
after = src[idx + len(marker):]
# The string starts with triple double-quote
delim = after[:3]  # should be '\"\"\"'
content_start = idx + len(marker) + 3
content = after[3:]
end_idx = content.rfind(delim)
extracted = content[:end_idx]
extracted = extracted.replace('\\\"\\\"\\\""', '\"\"\"')
with open('feature_selection_corrected.py', 'w', encoding='utf-8') as f:
    f.write(extracted)
import os
print('Written:', os.path.getsize('feature_selection_corrected.py'), 'bytes')
