module BertWordPieceTokenizer

# import dependencies
import HTTP
import IsURL

# export functions and variables
export init, tokenize, enable_truncation, encode, decode, token_to_id, id_to_token

# include local julia scripts
include("utils.jl")

# define global variables
# initialized vocabulary
global const _token2id = Ref{Dict{String, Int64}}(Dict{String, Int64}())
global const _id2token = Ref{Dict{Int64, String}}(Dict{Int64, String}())
global const _max_word_length = Ref{Int64}(200)
global const _do_lowercase = Ref{Bool}(false)
# max tokenized length
global const _max_length = Ref{Union{Int64, Nothing}}(nothing)
# truncation direction, post or pre
global const _truncation = Ref{String}("post")
# special tokens
global const _PAD = "[PAD]"
global const _UNK = "[UNK]"
global const _MASK = "[MASK]"
global const _CLS = "[CLS]"
global const _SEP = "[SEP]"
global const _SPECIAL_TOKENS = Vector{String}([_PAD, _UNK, _MASK, _CLS, _SEP])


function init(vocab_path::String; cache_path::Union{String, Nothing} = nothing, do_lowercase::Bool = false, max_word_length::Union{Int64, Nothing} = nothing)
    #= Initialize tokenizer
    Args:
      vocab_path: String, local path or http url to vocabulary
      cache_path: Union{String, Nothing}, specify the path to cache remote file in local
      do_lowercase: do lowercase
    =#
    global _token2id[] = load_vocab(vocab_path, cache_path=cache_path)
    global _id2token[] = Dict{Int64, String}((v, k) for (k, v) in _token2id[])
    global _do_lowercase[] = do_lowercase
    if max_word_length != nothing
        global _max_word_length[] = max_word_length
    end
end


function enable_truncation(max_length::Int64; truncation::String = "post")
    #= Enable truncation
    Args:
      max_length: Int64, specify max token length
      truncation: String, specify truncation from ["post", "pre"]
    =#
    @assert truncation in ["pre", "post"] "please specify truncation from [`pre`, `post`]"
    global _max_length[] = max_length
    global _truncation[] = truncation
end


function decode(token_ids::Vector{Int64})::String
    #= Recover text from token indices
    Args:
      token_ids: Vector{Int64}
    =#
    tokens = [id_to_token(id) for id in token_ids]
    tokens = [token for token in tokens if token ∉ _SPECIAL_TOKENS]
    text = ""
    for (i, token) in enumerate(tokens)
        if startswith(token, "##")
            text *= token[3:end]
        elseif length(token) == 1 && is_cjk_character(first(token))
            text *= token
        elseif length(token) == 1 && ispunct(first(token))
            text *= token * " "
        elseif i > 1 && is_cjk_character(first(text[end]))
            text *= token
        else
            text *= " " * token
        end
    end

    return strip(text)
end


function encode(first_text::String; second_text::Union{String, Nothing} = nothing)::Tuple{Vector{Int64}, Vector{Int64}}
    #= Encode text to token indices and segment indices
    Args:
      first_text: String, first text
      second_text: Union{String, Nothing} = nothing, second text
    =#
    tokens = tokenize(first_text, second_text=second_text)
    token_ids = []
    segment_ids = []
    segment_value = 0
    for token in tokens
        push!(token_ids, token_to_id(token))
        push!(segment_ids, segment_value)
        if token == _SEP
            segment_value = 1
        end
    end
    return token_ids, segment_ids
end


function tokenize(first_text::String; second_text::Union{String, Nothing} = nothing)::Vector{String}
    #= Tokenize text
    Args:
      first_text: String, first text
      second_text: Union{String, Nothing} = nothing, second text
    =#

    first_tokens = _tokenize(first_text)
    pushfirst!(first_tokens, _CLS)

    if second_text != nothing
        second_tokens = _tokenize(second_text)
        pushfirst!(second_tokens, _SEP)
        texts = [first_tokens, second_tokens]
    else
        texts = [first_tokens]
    end

    if _max_length[] != nothing
        max_length = _max_length[] - 1
        if _truncation[] == "post"
            offset = 0
        elseif _truncation[] == "pre"
            offset = 1
        end
        truncate_text!(texts, max_length, offset, _truncation[])
    end

    if second_text != nothing
        append!(first_tokens, second_tokens)
    end
    push!(first_tokens, _SEP)
    return first_tokens
end


function _tokenize(text::String)::Vector{String}
    #= Tokenize text
    Args:
      text: String, input text
    =#
    if _do_lowercase[]
        text = Base.lowercase(text)
    end

    tmp = ""
    for ch in text
        ch = first(ch)  # to Char
        if ispunct(ch) || is_cjk_character(ch)
            tmp *= " " * ch * " "
        elseif isspace(ch)
            tmp *= " "
        elseif Int64(ch) == 0 || Int64(ch) == 65533 || iscntrl(ch)
            continue
        else
            tmp *= ch
        end
    end

    tokens = Vector{String}()
    for word in split(strip(tmp))
        append!(tokens, _wordpiece_tokenize(String(word)))
    end

    return tokens
end


function _wordpiece_tokenize(word::String)::Vector{String}
    #= Word piece tokenize
    Args:
      word: String, a word
    =#
    if length(word) > _max_word_length[]
        return [word]
    end

    # to list of string
    word = split(word, "")

    tokens = Vector{String}()
    i = j = 0
    while i < length(word)
        j = length(word)
        sub = nothing
        while j > i
            sub = join(word[i+1:j])
            if i > 0
                sub = "##" * sub
            end
            if haskey(_token2id[], sub)
                break
            end
            j -= 1
        end
        if i == j
            j += 1
        end
        if sub != nothing
            push!(tokens, sub)
        end
        i = j
    end

    return tokens
end



function token_to_id(token::String)::Int64
    #= Convert token to the corresponding index
    Args:
      token: String
    =#
    if haskey(_token2id[], token)
        return _token2id[][token]
    end
    return _token2id[][_UNK]
end


function id_to_token(id::Int64)::String
    #= Convert index to the corresponding token
    Args:
      id: Int64
    =#
    if haskey(_id2token[], id)
        return _id2token[][id]
    end
    return _UNK
end


function load_vocab(vocab_path::String; cache_path::Union{String, Nothing} = nothing)::Dict{String, Int64}
    #= Load vocabulary from local file or remote http url
    Args:
      vocab_path: String, path to vocabulary
      cache_path: Union{String, nothing} = nothing, specify the path to cache remote file in local
    =#
    vocab = Dict{String, Int64}()

    if IsURL.isurl(vocab_path)
        println("downloading vocabulary from $vocab_path")
        r = HTTP.get(vocab_path)
        @assert r.status == 200
        lines = strip(String(r.body))
        if cache_path != nothing
            open(cache_path, "w") do writer
                write(writer, lines)
            end
        end
        for line in split(lines, "\n")
            token = split(line)
            if isempty(token)
                token = strip(line)
            else
                token = token[1]
            end
            vocab[token] = length(vocab)
        end
    else
        open(vocab_path) do reader
            for line in eachline(reader)
                token = split(line)
                if isempty(token)
                    token = strip(line)
                else
                    token = token[1]
                end
                vocab[token] = length(vocab)
            end
        end
    end
    return vocab
end


end # module
