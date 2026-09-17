# Book Recommendation System

A content-based book recommender that suggests similar titles using TF-IDF vectorization and cosine similarity — built on a public dataset of 11,127 books from Goodreads-style metadata.

## Problem Statement

Given a book title, recommend the 10 most similar books based on title and author text, ranked by similarity score.

## Dataset

- **Source:** `books_data.csv` — 11,127 books
- **Columns:** `bookID`, `title`, `authors`, `average_rating`
- No missing values across any column

## Approach

1. **Feature construction:** a `book_content` field is built by concatenating each book's `title` and `authors`
2. **Vectorization:** `TfidfVectorizer` (scikit-learn, English stopwords removed) converts every book's text into a TF-IDF vector
   - Resulting matrix: **11,127 × 17,937** (books × vocabulary terms)
   - Only **0.04%** of matrix cells are non-zero — a very sparse representation, as expected for short-text TF-IDF
3. **Similarity:** `linear_kernel` computes pairwise cosine similarity across all books, producing an **11,127 × 11,127** similarity matrix (~124 million similarity scores)
4. **Recommendation:** for a queried title, the top 10 most similar books (excluding itself) are returned, ranked by cosine similarity

## Sample Output

**Query:** *Harry Potter and the Order of the Phoenix*

| Rank | Recommended Title | Similarity |
|---|---|---|
| 1 | Harry Potter and the Chamber of Secrets | 0.782 |
| 2 | Harry Potter and the Sorcerer's Stone | 0.778 |
| 3 | Harry Potter and the Half-Blood Prince | 0.768 |
| 4 | Harry Potter and the Prisoner of Azkaban | 0.765 |
| 5 | Harry Potter Boxed Set, Books 1–5 | 0.764 |

For a series with a consistent author and title pattern, the model correctly clusters every other book in that series at the top, with similarity scores between 0.66 and 0.78.

**Query:** *Seven Plays*

| Rank | Recommended Title | Similarity |
|---|---|---|
| 1 | Buried Child | 0.458 |
| 2 | See You Around, Sam! | 0.263 |
| 3 | Sam Walton: Made In America | 0.246 |

## Key Observations / Limitations

- Because `book_content` is built only from **title + author** (no genre, description, or subject metadata), recommendation quality depends heavily on how much textual overlap exists between titles/author names. Multi-book series with a consistent author (like Harry Potter) get strong, coherent recommendations (0.66–0.78 similarity).
- For less "textually distinctive" queries — e.g. *Seven Plays* — the model can surface loosely related matches driven by incidental word overlap (like books written by *any* author named "Sam"), rather than genuine thematic similarity. This is the main trade-off of a pure title/author TF-IDF approach vs. one enriched with genre or description text.
- This is a **content-based** recommender (similarity driven by item text), not collaborative filtering (which would need a user–item ratings matrix). A natural next step would be adding genre/description fields to the content string, or layering in collaborative filtering if user rating histories become available.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, scikit-learn (`TfidfVectorizer`, `linear_kernel`), Plotly (for exploratory rating distribution/author charts)

## How to Run

```bash
pip install pandas numpy scikit-learn plotly
python book_recommendation_system.py
```

## What I Learned

This project was my first hands-on build of a similarity-based recommender — going from raw text to a TF-IDF matrix to a full pairwise cosine similarity matrix, and using it to serve ranked recommendations. The more useful lesson came from testing edge cases: the model's occasional weak matches on generic queries taught me that a recommender is only as good as the features it's built on, and that "content-based" needs genuinely descriptive content — title and author alone can be too thin a signal once you look past the easy cases like well-known series.
