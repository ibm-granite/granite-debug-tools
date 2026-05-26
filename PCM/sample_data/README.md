# Sample Data

Multi-turn conversations between a user and an AI assistant, used to demonstrate the PCM knowledge extraction pipeline.

## conversations.json

12 conversations covering diverse personal information domains:

| ID | Turns | Topics |
|----|-------|--------|
| sample_01 | 6 | Recipes, health, travel, pets, hobbies, food |
| sample_02 | 6 | Family trip, travel, hobbies, food |
| sample_03 | 6 | Travel spending, hobbies, purchases, food |
| sample_04 | 6 | Pets, work, purchases, food subscriptions |
| sample_05 | 6 | Travel, pets, purchases, food/restaurants |
| sample_06 | 3 | Family, travel, work |
| sample_07 | 10 | Travel preferences, pets, work, food |
| sample_08 | 7 | Work, purchases, food/baking |
| sample_09 | 6 | Concerts, family, health, travel, pets, purchases |
| sample_10 | 6 | Relationships, pets, hobbies, work, food |
| sample_11 | 6 | Family, travel, work, purchases, food |
| sample_12 | 6 | Car issues, family, travel, pets, work |

### Format

```json
[
  {
    "conversation_id": "sample_01",
    "source_id": "answer_07664d43_1",
    "description": "Recipes, health, travel, pets, hobbies, food",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

### Source

Conversations are derived from the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark dataset, selected for diversity of personal information content across domains such as family, health, travel, work, hobbies, purchases, and food.
