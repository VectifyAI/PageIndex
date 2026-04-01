

## Document Search by Metadata
<callout>PageIndex with metadata support is in closed beta. Fill out this form to request early access to this feature.</callout>

For documents that can be easily distinguished by metadata, we recommend using metadata to search the documents.
This method is ideal for the following document types:
- Financial reports categorized by company and time period
- Legal documents categorized by case type
- Medical records categorized by patient or condition
- And many others

In such cases, you can search documents by leveraging their metadata. A popular method is to use "Query to SQL" for document retrieval.


### Example Pipeline

#### PageIndex Tree Generation
Upload all documents into PageIndex to get their `doc_id`.

#### Set up SQL tables

Store documents along with their metadata and the PageIndex `doc_id` in a database table.

#### Query to SQL

Use an LLM to transform a user’s retrieval request into a SQL query to fetch relevant documents.

#### Retrieve with PageIndex

Use the PageIndex `doc_id` of the retrieved documents to perform further retrieval via the PageIndex retrieval API.

## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/meB40zV0)