import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, My name is saurabh"
tokens = enc.encode(text)

print("Tokens:", tokens)

TokensEncoded= [13225, 11, 3673, 1308, 382, 96446, 25482]

decode = enc.decode(TokensEncoded)
print("Decode: " , decode)