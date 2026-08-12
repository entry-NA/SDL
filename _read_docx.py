import docx
path = r'C:\Users\23991\Desktop\副本讨论大纲.docx'
print(f"Trying to read: {path}")
import os
print(f"File exists: {os.path.exists(path)}")
doc = docx.Document(path)
for p in doc.paragraphs:
    t = p.text.strip()
    if t:
        print(t)