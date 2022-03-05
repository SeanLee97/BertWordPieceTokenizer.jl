function is_cjk_character(ch::Union{Char, Int64})::Bool
    #= check wheter a character is cjk charachter
    Args:
      ch: Union{Char, Int64}, a character or an integer
    ref: https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    =#
    ch = Int64(ch)
    #return (0x4E00 <= ch <= 0x9FFF) || (0x3400 <= ch <= 0x4DBF) || (0x20000 <= ch <= 0x2A6DF) || (0x2A700 <= ch <= 0x2B73F) || (0x2B740 <= ch <= 0x2B81F) || (0x2B820 <= ch <= 0x2CEAF) || (0xF900 <= ch <= 0xFAFF) || (0x2F800 <= ch <= 0x2FA1F)
    return (19968 <= ch <= 40959) || (13312 <= ch <= 19903) || (131072 <= ch <= 173791) || (173824 <= ch <= 177983) || (177984 <= ch <= 178207) || (178208 <= ch <= 183983) || (63744 <= ch <= 64255) || (194560 <= ch <= 195103)
end


function truncate_text!(texts::Union{Vector{String}, Vector{Vector{String}}}, max_length::Int64, offsets::Union{Int64, Vector{Int64}}, truncation::String)::Union{Vector{String}, Vector{Vector{String}}}
    #= truncate text
    Args:
      texts: Union{Vector{String}, Vector{Vector{String}}}, tokenized text or list of tokenized text
      max_length: Int64, max tokens length
      offsets: Union{Int64, Vector{Int64}}, offsets of texts to delete
      truncation: String, `post` or `pre`
    =#

    @assert truncation in ["pre", "post"] "please specify direction from [`pre`, `post`]"

    if isa(texts, String)
        texts = [texts]
    end

    if isa(offsets, Int64)
        offsets = [offsets for _ in 1:length(texts)]
    end

    while true
        lengths = [length(s) for s in texts]
        if sum(lengths) > max_length
            i = argmax(lengths)
            base_index = truncation == "post" ? lastindex(texts[i]) : firstindex(texts[i])
            deleteat!(texts[i], base_index + offsets[i])
        else
            return texts
        end
    end
end
