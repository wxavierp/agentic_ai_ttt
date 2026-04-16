"""
chromadb_skeleton.py
ChromaDB — Persistent Vector Store
Capstone Prep · Complete the 4 TODOs below

Run:     python chromadb_skeleton.py   (or paste cells into Colab)
Install: pip install openai chromadb python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb

load_dotenv()
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version    = "2024-08-01-preview",
)
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-ada-002")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Open (or create) a persistent store
#
# PersistentClient writes everything to a folder on disk.
# Next time you run this file, the data is still there — no re-seeding needed.
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "./my_tax_db"

chroma     = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(
    name     = "tax_knowledge",
    metadata = {"hnsw:space": "cosine"},
)

print(f"Store opened at : {DB_PATH}")
print(f"Records stored  : {collection.count()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — embed() helper  (provided — do not change)
# ─────────────────────────────────────────────────────────────────────────────

def embed(text):
    """Convert text to a vector using the Azure embeddings API."""
    return client.embeddings.create(
        model=EMBED_MODEL, input=text.replace("\n", " ")
    ).data[0].embedding


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Seed the store  (provided — runs only when collection is empty)
# ─────────────────────────────────────────────────────────────────────────────

TAX_SECTIONS = [
    {
        "id":   "igst_5",
        "text": "IGST Act Section 5: Inter-state supply of services attracts integrated "
                "tax at 18% for IT and software services.",
        "meta": {"act": "IGST", "section": "5", "topic": "inter-state"},
    },
    {
        "id":   "igst_16",
        "text": "IGST Act Section 16(1): Zero-rated supply includes export of services. "
                "LUT must be filed in Form GST RFD-11 to export at 0% rate.",
        "meta": {"act": "IGST", "section": "16", "topic": "export"},
    },
    {
        "id":   "cgst_9",
        "text": "CGST Act Section 9: Intra-state software services attract 18% "
                "split as CGST 9% and SGST 9%.",
        "meta": {"act": "CGST", "section": "9", "topic": "intra-state"},
    },
    {
        "id":   "cgst_9_3",
        "text": "CGST Act Section 9(3): Legal services by an advocate to a business "
                "entity are taxed under reverse charge mechanism.",
        "meta": {"act": "CGST", "section": "9(3)", "topic": "rcm"},
    },
    {
        "id":   "schedule3",
        "text": "CGST Schedule III: Healthcare services by clinical establishments and "
                "educational services by educational institutions are exempt from GST.",
        "meta": {"act": "CGST", "section": "schedule3", "topic": "exempt"},
    },
]

if collection.count() == 0:
    print("\nSeeding knowledge base (first run only)...")
    for s in TAX_SECTIONS:
        collection.add(
            ids=[s["id"]], documents=[s["text"]],
            embeddings=[embed(s["text"])], metadatas=[s["meta"]],
        )
        print(f"  Added [{s['id']}]")
    print(f"Done. {collection.count()} records in {DB_PATH}")
else:
    print("Knowledge base already seeded.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — search() helper  (provided — do not change)
# ─────────────────────────────────────────────────────────────────────────────

def search(query_text, n_results=3, where=None):
    """
    Find the most similar stored documents to query_text.
    Optional `where` adds a metadata filter:  where={"act": {"$eq": "IGST"}}

    Returns: [{"id": ..., "text": ..., "metadata": ..., "similarity": ...}]
    """
    q_vec  = embed(query_text)
    kwargs = {"query_embeddings": [q_vec],
              "n_results": min(n_results, collection.count())}
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    return [
        {"id":       res["ids"][0][i],
         "text":     res["documents"][0][i],
         "metadata": res["metadatas"][0][i],
         "similarity": round(1 - res["distances"][0][i], 3)}
        for i in range(len(res["ids"][0]))
    ]


def show(label, results):
    """Pretty-print search results."""
    print(f'\nQuery: "{label}"')
    print("-" * 60)
    for r in results:
        print(f"  [{r['id']}]  sim={r['similarity']}  {r['text'][:70]}...")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Try the provided searches first, then attempt the TODOs
# ─────────────────────────────────────────────────────────────────────────────

show("software invoice Mumbai to Bengaluru",
     search("software invoice Mumbai to Bengaluru"))

show("export foreign client zero tax",
     search("export foreign client zero tax"))

show("IGST act only (metadata filter)",
     search("tax on services", where={"act": {"$eq": "IGST"}}))


# =============================================================================
#
#  TODO 1 — Write your own semantic search query
#
#  Change the string below and re-run.
#  Try: "advocate legal services charges"
#       "healthcare hospital GST exempt"
#       "government road construction contract"
#
# =============================================================================

YOUR_QUERY = "YOUR QUERY HERE"   # ← change this string

if YOUR_QUERY != "YOUR QUERY HERE":
    show(YOUR_QUERY, search(YOUR_QUERY))


# =============================================================================
#
#  TODO 2 — Add a new document to the store
#
#  Add the GSTR-1 filing requirement as a new section.
#  After adding it, search for "GSTR-1 return filing deadline" and confirm
#  it appears at the top of the results.
#
#  Steps:
#    1. Build the new_section dict (id, text, meta)
#    2. Check it doesn't already exist:
#         existing = collection.get(ids=["gstr1"])
#         if not existing["ids"]:  # empty → safe to add
#    3. Call collection.add(ids=[...], documents=[...],
#                           embeddings=[embed(...)], metadatas=[...])
#    4. Print collection.count() — should be 6 now
#    5. Run: show("GSTR-1 filing", search("GSTR-1 return filing deadline"))
#
# =============================================================================

# YOUR CODE HERE


# =============================================================================
#
#  TODO 3 — Filter with $ne (not equal)
#
#  Find all sections that are NOT about exempt supplies.
#  Then find all CGST sections only.
#
#  Hint: where={"topic": {"$ne": "exempt"}}
#        where={"act":   {"$eq": "CGST"}}
#
# =============================================================================

# YOUR CODE HERE
# show("non-exempt sections",   search("GST rate",   where={"topic": {"$ne": "exempt"}}))
# show("CGST sections only",    search("tax rate",   where={"act":   {"$eq": "CGST"}}))


# =============================================================================
#
#  TODO 4 — Delete a record and verify it is gone
#
#  Delete "schedule3" (the exempt healthcare section).
#  Then search for "healthcare exempt" — it should no longer appear.
#  Then restore it (re-add it) so the store is complete again.
#
#  Hint: collection.delete(ids=["schedule3"])
#
# =============================================================================

# YOUR CODE HERE
# collection.delete(ids=["schedule3"])
# print(f"\nAfter delete: {collection.count()} records")
# show("healthcare (should not return schedule3)", search("healthcare exempt"))
# ... then re-add it


# ─────────────────────────────────────────────────────────────────────────────
# Inspect — show everything currently in the store
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\nAll {collection.count()} records in the store:")
all_data = collection.get()
for doc_id, text, meta in zip(
    all_data["ids"], all_data["documents"], all_data["metadatas"]
):
    print(f"  [{doc_id}]  act={meta.get('act','-'):<6}  topic={meta.get('topic','-'):<12}  {text[:65]}...")
