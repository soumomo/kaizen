PRE_TOKEN_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re 
from multiprocessing import Pool
import os

def pretokenize_chunk(chunk , special_tokens):
    '''process one chunk - runs in parallel'''
    word_frequencies = {}

    # split on special tokens so they never get merged into pretokenization
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in sorted_specials)
        segments = re.split(f"({pattern})", chunk)
    else:
        segments = [chunk]

    for segment in segments:
        if not segment:
            continue
        if segment in special_tokens:
            # special tokens never participate in pretokenization/merges
            continue
        for match in re.finditer(PRE_TOKEN_REGEX , segment):
            word = match.group()
            word_bytes  = tuple(word.encode("utf-8"))
            word_frequencies[word_bytes] = word_frequencies.get(word_bytes , 0) + 1
    return word_frequencies

# split corpus into chunk at document boundaries
def find_chunk_boundaries(file , num_chunks , boundary_token):
    file_size = file.seek(0 , os.SEEK_END)
    num_chunks = max(1, min(num_chunks, file_size))  # never more chunks than bytes
    chunk_size = file_size//num_chunks
    boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    boundaries[-1] = file_size  # guarantee correctness regardless of rounding

    
    #adjust vboundaries to lanf on special tokens (document boundaries)
    for i in range(1 , len(boundaries) - 1):
        file.seek(boundaries[i])
        offset = 0

        # search for next occurence of boundary token
        while True:
            chunk = file.read(4096)
            if not chunk:
                boundaries[i] = file_size
                break

            pos = chunk.find(boundary_token)
            if pos != -1:
                boundaries[i] += offset + pos
                break
            offset += len(chunk)

    boundaries = sorted(set(boundaries))
    return boundaries

def train_bpe_tokenizer(input_path , vocab_size , special_tokens , num_processes = None):
    if num_processes is None:
        num_processes = os.cpu_count()

        #split corpus into chunks
    with open(input_path , "rb") as f:
        boundaries = find_chunk_boundaries(f , num_processes*3 , b"<|endoftext|>")
        chunks = []
        for start , end in zip(boundaries[:-1] , boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end-start).decode("utf-8"))

    # parallel pre-tokenization
    with Pool(num_processes) as pool:
        chunk_frequencies = pool.starmap(
            pretokenize_chunk,
            [(chunk , special_tokens) for chunk in chunks]

        )

    #combine word frequncies from al chunks
    word_frequencies = {}

    for chunk_freq in chunk_frequencies:
        for word, freq in chunk_freq.items():
            word_frequencies[word] = (
                word_frequencies.get(word, 0) + freq
            )


    #initialize vocabulary with all byte values
    vocab = {idx: bytes([idx]) for idx in range(256)}

    #add special tokens
    for i , token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    
    merges = []
    num_merges = vocab_size - len(vocab)
    #count all the adjacent pairs across all words once
    pair_frequencies = {}
    
    def get_pairs(word):
        return [(word[i] , word[i+1]) for i in range(len(word) - 1)]
    
    for word , freq in word_frequencies.items():
        for pair in get_pairs(word):
            pair_frequencies[pair] = pair_frequencies.get(pair , 0) + freq

# Now the merge loop only updates what changes:

    for _ in range(num_merges):
        if not pair_frequencies:
            break
            
        # find the most frequent pair (tie-breakoing lexicographically)

        best_pair = max(
            pair_frequencies.keys() , 
            key = lambda p: (pair_frequencies[p] , vocab[p[0]] , vocab[p[1]])
        )
            
        #create new token
        new_id = len(vocab)
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # apply merge to all words
        new_word_frequencies = {}

        for word , freq in word_frequencies.items():
            if best_pair not in get_pairs(word):
                #word is unchanged so just copy it
                new_word_frequencies[word] = freq
            else:
                #subtract old pair counts
                for pair in get_pairs(word):
                    pair_frequencies[pair] -= freq
                    if pair_frequencies[pair] == 0:
                        del pair_frequencies[pair]
                
                #apply the merge
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_frequencies[tuple(new_word)] = freq

                for pair in get_pairs(new_word):
                    pair_frequencies[pair] = pair_frequencies.get(pair , 0) + freq


        word_frequencies = new_word_frequencies

    return vocab , merges



class Tokenizer:
    def __init__(self , vocab: dict ,merges: list , special_tokens: list = None):
         self.decoder = vocab # {idx: bytes([idx]) for idx in range(256)}
        # reverses {idx: bytes} into {bytes: idx}
         self.encoder = {bytes_val :idx for idx , bytes_val in vocab.items()}
         self.merges = merges
         
         self.special_tokens = special_tokens or []

         if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_specials)
            self.special_pattern = re.compile(f"({pattern})")
         else:
            self.special_pattern = None
             


    def decode(self , ids: list[int]) -> str:
        byte_chunks = [self.decoder[idx] for idx in ids]
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8" , errors ="replace")    

    def encode(self , text: str) -> list[int]:
        if self.special_tokens:
            segments = self.special_pattern.split(text)
        else:
            segments = [text]

        final_ids = []
        
        for segment in segments:
            if not segment:
                continue
                
            # check if this specific segment is a special token
            if segment in self.special_tokens:
                final_ids.append(self.encoder[segment.encode("utf-8")])
                continue  
            
            # If it is not a special token, process it as normal text
            for match in re.finditer(PRE_TOKEN_REGEX, segment):
                word = match.group()
                word_ids = [self.encoder[bytes([b])] for b in word.encode("utf-8")]

            
                for p0 , p1 in self.merges:
                    combined_bytes = p0 + p1
                    new_id = self.encoder[combined_bytes]

                    id0 = self.encoder[p0]
                    id1 = self.encoder[p1]
                    pair_to_find = (id0 , id1)


                    #scan and replace the pair with new ID
                    new_word = []
                    i = 0
                    while i < len(word_ids):
                        if i < len(word_ids) - 1 and (word_ids[i], word_ids[i + 1]) == pair_to_find:
                            new_word.append(new_id)
                            i += 2
                        else:
                            new_word.append(word_ids[i])
                            i += 1
                    word_ids = new_word

                final_ids.extend(word_ids)


        return final_ids
                



    def encode_iterable(self, texts):
        """
        > accepts an iterable of strings (like lines in a file).
        > yields token IDs one by one dynamically (as a Python generator) instead of storing the whole file's tokens in memory all at once.
        """
        for text in texts:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id



        
    
    
        
    
