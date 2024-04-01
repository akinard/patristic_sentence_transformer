#!/usr/bin/env python

"""
This script performs sentence transform cosine analysis on libraries of patristic texts.

Copyright (c) 2024  Andrew Kinard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



# TODO: add any needed sources to the Palamas_sources and the Patristic_sources directories

# Requirements
# pip install sentence-transformers

import os
import argparse
import sys
import itertools
from sentence_transformers import SentenceTransformer, util
import numpy as np
# import polars as pl
import pandas as pd
import concurrent.futures
import threading
import time

# Load a pre-trained model
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare documents in two directories.')
parser.add_argument('dir1', type=str, help='Path to the first directory')
parser.add_argument('dir2', type=str, help='Path to the second directory')
parser.add_argument('parquet_file', type=str, help='Path to the parquet file')
args = parser.parse_args()

# Load or create the DataFrame
if os.path.exists(args.parquet_file):
    df = pd.read_parquet(args.parquet_file)
else:
    df = pd.DataFrame(
        columns=[
            "doc1_path",
            "sentence1",
            "doc2_path",
            "sentence2",
            "max_similarity",
            "hit"
        ]
    )

def read_file(filepath):
    """Reads a text file and returns a list of sentences."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = [sentence.strip() for sentence in text.split('.') if len(sentence.split()) >= 10]

    return sentences

# Define the threshold as a global variable
global_threshold = 0.50

def compare_documents(doc1_path, doc2_path):
    # Read and encode documents
    doc1_sentences = read_file(doc1_path)
    doc2_sentences = read_file(doc2_path)
    
    # Compute embeddings for both documents
    doc1_embeddings = model.encode(doc1_sentences)
    doc2_embeddings = model.encode(doc2_sentences)
    
    # List to store all pairs of sentences with a similarity score higher than the threshold
    similar_sentences = []

    # Compare each sentence in doc1 to all sentences in doc2
    for sentence1, embedding1 in zip(doc1_sentences, doc1_embeddings):
        similarities = util.cos_sim(embedding1, doc2_embeddings)[0].cpu().numpy()
        
        # Find the maximum similarity score for the current sentence in doc1 across all sentences in doc2
        max_similarity = np.max(similarities)
        
        if max_similarity > global_threshold:
            max_index = np.argmax(similarities)
            sentence2 = doc2_sentences[max_index]
            print(f"Found similar ideas:\n-[{doc1_path}] {sentence1}\n-[{doc2_path}] {sentence2}\nSimilarity: {max_similarity}\n")
            
            # Add the pair of sentences and the similarity score to the list
            similar_sentences.append((sentence1, sentence2, max_similarity))

    # Return the list of all pairs of sentences with a similarity score higher than the threshold
    return similar_sentences


def compare_directories(dir1, dir2):
    global df
    files_1 = [os.path.join(root, file) for root, dirs, files in os.walk(dir1) for file in files]
    files_2 = [os.path.join(root, file) for root, dirs, files in os.walk(dir2) for file in files]
    
    total_files = len(files_1) * len(files_2)
    processed_files = 0

    for file1, file2 in itertools.product(files_1, files_2):
        # Skip if this combination of files already exists in the DataFrame
        if ((df['doc1_path'] == file1) & (df['doc2_path'] == file2)).any():
            print(f"Skipping [{file1}] and [{file2}]. Already exists in the DataFrame.")
            processed_files += 1
            continue

        try:
            similar_sentences = compare_documents(file1, file2)
            if similar_sentences:
                for sentence1, sentence2, max_similarity in similar_sentences:
                    new_row = {
                            "doc1_path": file1,
                            "sentence1": sentence1,
                            "doc2_path": file2,
                            "sentence2": sentence2,
                            "max_similarity": max_similarity,
                            "hit": "1"
                        }
                    df.loc[len(df)] = new_row
            else:
                new_row = {
                        "doc1_path": file1,
                        "sentence1": "No similar sentence found",
                        "doc2_path": file2,
                        "sentence2": "No similar sentence found",
                        "max_similarity": 0,
                        "hit": "0"
                    }
                df.loc[len(df)] = new_row

            # Increment the count of processed files
            processed_files += 1

            # Calculate and print the percentage completion
            completion_percentage = (processed_files / total_files) * 100
            print(f"Completion: {completion_percentage:.2f}%")
        except Exception as e:
            print(f"An error occurred while comparing [{file1}] and [{file2}]: {str(e)}")


def save_df_every_5_minutes():
    global df  # Add this line to declare df as a global variable
    while True:
        try:
            # Save the DataFrame back into the parquet file
            df.to_parquet(args.parquet_file)
            print("DataFrame saved by timer.")
        except Exception as e:
            print(f"An error occurred while saving the DataFrame: {str(e)}")
        
        # Wait for 5 minutes
        time.sleep(5 * 60)

if __name__ == "__main__":
    try:
        # Start a new thread that saves the DataFrame every 5 minutes
        threading.Thread(target=save_df_every_5_minutes, daemon=True).start()

        # Call the function with your directories
        compare_directories(args.dir1, args.dir2)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving DataFrame and exiting...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Remove duplicate rows
        df = df.drop_duplicates()

        # Save the DataFrame back into the parquet file
        df.to_parquet(args.parquet_file)
        print("DataFrame saved on exit.")
