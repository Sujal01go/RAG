from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Sample Tesla text with structured sections
tesla_text = """Tesla reported record revenue of $25.8B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 46% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


# Semantic Chunker - groups by meaning, not structure
# Using a local Hugging Face embedding model instead of OpenAI
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)


# Split the text into semantic chunks
chunks = semantic_splitter.split_text(tesla_text)


# Display the results
print("SEMANTIC CHUNKING RESULTS")
print("=" * 50)

for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Length: {len(chunk)} characters")
    print("Content:")
    print(chunk)
    print("-" * 50)

print(f"\nTotal chunks created: {len(chunks)}")
