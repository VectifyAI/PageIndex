## Document Search by Semantics

For documents that cover diverse topics, one can also use vector-based semantic search to search the documents. The procedure is slightly different from the classic vector-search-based method.

### Example Pipeline


#### Chunking and Embedding
Divide the documents into chunks, choose an embedding model to convert the chunks into vectors and store each vector with its corresponding `doc_id` in a vector database.


#### Vector Search

For each query, conduct a vector-based search to get top-K chunks with their corresponding documents. 

#### Compute Document Score

For each document, calculate a relevance score. Let N be the number of content chunks associated with each document, and let **ChunkScore**(n) be the relevance score of chunk n. The document score is computed as:


$$
\text{DocScore}=\frac{1}{\sqrt{N+1}}\sum_{n=1}^N \text{ChunkScore}(n)
$$

- The sum aggregates relevance from all related chunks.
- The +1 inside the square root ensures the formula handles nodes with zero chunks.
- Using the square root in the denominator allows the score to increase with the number of relevant chunks, but with diminishing returns. This rewards documents with more relevant chunks, while preventing large nodes from dominating due to quantity alone.
- This scoring favors documents with fewer, highly relevant chunks over those with many weakly relevant ones.


#### Retrieve with PageIndex

Select the documents with the highest DocScore, then use their `doc_id` to perform further retrieval via the PageIndex retrieval API.



## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/meB40zV0)