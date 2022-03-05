using Test
import BertWordPieceTokenizer as BWP


# initialize tokenizer
BWP.init("https://huggingface.co/bert-base-uncased/raw/main/vocab.txt", do_lowercase=true)

@test BWP.tokenize("I like apples") == ["[CLS]", "i", "like", "apples", "[SEP]"]
@test BWP.encode("I like apples") == (Vector{Int64}([101, 1045, 2066, 18108, 102]), Vector{Int64}([0, 0, 0, 0, 0]))
@test BWP.decode(Vector{Int64}([101, 1045, 2123, 2102, 2066, 18108, 102])) == "i dont like apples"
@test BWP.tokenize("I like apples", second_text="I dont like apples") == ["[CLS]", "i", "like", "apples", "[SEP]", "i", "don", "##t", "like", "apples", "[SEP]"]
@test BWP.encode("I like apples", second_text="I dont like apples") == (Vector{Int64}([101, 1045, 2066, 18108, 102, 1045, 2123, 2102, 2066, 18108, 102]), Vector{Int64}([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))

BWP.enable_truncation(5)
@test BWP.tokenize("I like apples", second_text="I dont like apples") == ["[CLS]", "i", "[SEP]", "i", "[SEP]"]

BWP.enable_truncation(5, truncation="pre")
@test BWP.tokenize("I like apples", second_text="I dont like apples") == ["[CLS]", "apples", "[SEP]", "apples", "[SEP]"]
