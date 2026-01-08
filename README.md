# Roommate Matching Agent Service

Cost-safe Python FastAPI service for BAABA.ng roommate matching with optional LlamaIndex AI agent.

## Architecture

**Hybrid Approach for Cost Safety:**
- **90% of requests:** Direct Python scoring (no LLM call, free, fast)
- **10% of requests:** LlamaIndex agent for natural language queries (costs $0.01-0.05)

## Running Locally

### Prerequisites
- Python 3.11+
- pip or conda

### Setup

1. **Create virtual environment:**
```bash
cd services/roommate_agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - SUPABASE_URL (from main project .env)
# - SUPABASE_SERVICE_ROLE_KEY (from Supabase dashboard)
# - OPENAI_API_KEY (from OpenAI platform)
```

4. **Run server:**
```bash
uvicorn app:app --reload --port 8001
```

5. **Test:**
```bash
# Visit http://localhost:8001/docs for Swagger UI
# Or test with curl:
curl -X POST http://localhost:8001/roommate-matching \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "<test_user_uuid>",
    "limit": 10,
    "min_score": 60
  }'
```

## API Endpoints

### `POST /roommate-matching`

Primary matching endpoint.

**Request:**
```json
{
  "user_id": "uuid",
  "limit": 20,
  "min_score": 60,
  "filters": {
    "budget_max": 50000,
    "location": "Lagos",
    "mbti_types": ["INTJ", "INFP"],
    "min_date": "2025-01-01",
    "max_date": "2025-03-01"
  },
  "query": "Find roommates who like Afrobeats"  // OPTIONAL: triggers agent
}
```

**Response:**
```json
{
  "matches": [
    {
      "match_id": "user1_user2",
      "target_user": { ... },
      "compatibility_score": 78.5,
      "compatibility_breakdown": {
        "mbti_compatibility": 20,
        "lifestyle_match": 22,
        "budget_alignment": 15,
        "location_match": 10,
        "date_compatibility": 8,
        "tolerance_match": 10,
        "interests_overlap": 3,
        "music_taste": 7
      },
      "created_at": "2025-12-23T12:00:00Z"
    }
  ],
  "total_eligible_users": 50,
  "algorithm_version": "v2.0.0-direct"
}
```

### `GET /health`
Health check endpoint

### `GET /admin/stats`
Cache and usage statistics

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | (required) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key | (required) |
| `OPENAI_API_KEY` | OpenAI API key | (required if using agent) |
| `OPENAI_MODEL` | Model name | `gpt-4-turbo-preview` |
| `USE_AGENT_BY_DEFAULT` | Always use agent | `false` |
| `MAX_CANDIDATES_FOR_AGENT` | Limit candidates for agent | `50` |
| `ENABLE_RESULT_CACHING` | Enable caching | `true` |
| `CACHE_TTL_MINUTES` | Cache time-to-live | `5` |
| `RATE_LIMIT_PER_MINUTE` | Rate limit | `10` |
| `FALLBACK_TO_DIRECT_ON_ERROR` | Fallback if agent fails | `true` |
| `PORT` | Server port | `8001` |
| `CORS_ORIGINS` | Allowed origins (JSON array) | `["http://localhost:5173"]` |

## Cost Control

**Default behavior (no query):**
- Uses direct Python scoring
- No OpenAI API calls
- Free, fast, predictable
- Same results as old Edge Function

**With natural language query:**
- Uses LlamaIndex agent
- ~$0.01-0.05 per request
- Enable features like semantic search

**Monthly cost estimate:**
- 1000 users × 5 searches = 5000 requests
- 90% direct (free) = 4500 × $0 = $0
- 10% agent = 500 × $0.02 = $10
- **Total: ~$10/month**

**Safeguards:**
1. Set OpenAI monthly budget limit in dashboard
2. Rate limiting (10 req/min per IP)
3. Result caching (5 minutes)
4. Automatic fallback to direct if agent fails

## File Structure

```
services/roommate_agent/
├── app.py              # FastAPI application & endpoints
├── supabase_client.py  # Database client wrapper
├── compatibility.py    # Pure scoring functions (ported from TS)
├── agent.py            # LlamaIndex agent (future)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
└── README.md           # This file
```

## Deployment

### Local Testing
1. Run service: `uvicorn app:app --reload --port 8001`
2. Test endpoint: `curl http://localhost:8001/health`
3. Update Edge Function env: `ROOMMATE_AGENT_URL=http://localhost:8001`

### Production
1. Deploy to cloud (Heroku, Railway, Render, or Docker)
2. Set production `ROOMMATE_AGENT_URL` in Supabase Edge Functions
3. Monitor OpenAI dashboard for costs
4. Set alerts at 50%, 75%, 90% of budget

## Development Notes

- **Scoring functions are pure:** No side effects, deterministic results
- **Agent is opt-in:** Only used when query provided
- **Backward compatible:** Response matches old Edge Function exactly
- **Reuses env patterns:** Follows same config style as TS codebase
- **Music data:** Uses Spotify profiles only, ignores manual selections

## Testing

```bash
# Create test user profiles in Supabase
# Test direct matching (no agent)
curl -X POST http://localhost:8001/roommate-matching \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-uuid", "limit": 10}'

# Test with query (agent)
curl -X POST http://localhost:8001/roommate-matching \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-uuid", "query": "Find Afrobeats lovers"}'

# Check stats
curl http://localhost:8001/admin/stats
```

## Next Steps

1. Implement `agent.py` with LlamaIndex tools (cur Rently falls back to direct)
2. Add embeddings for semantic profile search
3. Add OpenAI usage tracking and alerts
4. Upgrade cache to Redis for multi-instance deployments
