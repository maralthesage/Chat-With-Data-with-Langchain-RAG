#!/usr/bin/env python3
"""
Debug Script for CSV RAG Analyst
===============================

This script helps debug issues with the RAG system by providing detailed output
of each processing step.

Usage:
    python debug_rag.py "your question here"
    
Example:
    python debug_rag.py "how many orders had ART_NR 086L06P in them?"
"""

import sys
from rag_engine.loader import load_and_prepare_csv
from rag_engine.analyzer import OllamaCsvRAG
from config import rechnung_path

def debug_rag_question(question: str):
    """Debug a specific question through the RAG system."""
    print("🔧 CSV RAG Debug Mode")
    print("=" * 50)
    
    # Load data
    print("📊 Loading CSV data...")
    try:
        df = load_and_prepare_csv(rechnung_path)
        print(f"✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"📅 Date range: {df['DATUM'].min()} to {df['DATUM'].max()}")
        print(f"🧾 Available columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize RAG with debug mode
    print("\n🤖 Initializing RAG system...")
    rag = OllamaCsvRAG(df, debug=True)
    
    # Process question
    print(f"\n❓ Processing question: '{question}'")
    try:
        answer = rag.ask(question)
        print(f"\n✅ FINAL ANSWER:\n{answer}")
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_rag.py \"your question here\"")
        print("Example: python debug_rag.py \"how many orders had ART_NR 086L06P in them?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    debug_rag_question(question)

if __name__ == "__main__":
    main()
