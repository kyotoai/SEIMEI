Topic: {topic}
Sample index: {sample_index} of {total_samples}

# Goal
Create one web-search evaluation question for the topic. The question must require
multiple web searches and synthesis across sources.

# Question requirements
- Explicitly instruct the assistant to use web search.
- Require at least two distinct sources (different domains).
- Require at least one concrete number (date, percent, count, amount).
- Require a short comparison or conclusion based on the sources.

# Answer scoring rubric
- Output `answer_scoring` as a JSON array of objects with:
  - requirement (string)
  - score (integer points)
- Provide 4-6 requirements.
- The scores must sum to 10.
- Include at least one requirement about source citations with URLs,
  one about numeric detail, and one about synthesis/comparison.

# Style
- Keep the question concise and specific.
- Vary the phrasing across samples so the dataset is not repetitive.
- Avoid requests for sensitive advice (medical, legal, or political guidance).

# Output
Return only the JSON object described in the system prompt.
