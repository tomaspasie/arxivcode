"""Test script for the /explain API endpoint"""
import requests
import json
import os


def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_explain_endpoint():
    """Test the /explain endpoint with example data"""
    print("\n=== Testing /explain Endpoint ===")

    # Test case 1: Contrastive Learning
    test_request = {
        "query": "contrastive learning",
        "code_snippet": """def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    similarity = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(similarity, labels)""",
        "paper_title": "SimCLR: A Simple Framework for Contrastive Learning of Visual Representations",
        "paper_context": "This paper introduces SimCLR, a framework for contrastive self-supervised learning using normalized temperature-scaled cross entropy loss (NT-Xent)."
    }

    try:
        response = requests.post(
            "http://localhost:8000/explain",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nQuery: {result['query']}")
            print(f"Paper: {result['paper_title']}")
            print(f"Model: {result['model']}")
            print(f"\nExplanation:\n{result['explanation']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_multiple_examples():
    """Test with multiple different examples"""
    print("\n=== Testing Multiple Examples ===")

    test_cases = [
        {
            "query": "attention mechanism",
            "code_snippet": """def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v), attn_weights""",
            "paper_title": "Attention Is All You Need",
            "paper_context": "Introduces the Transformer architecture with multi-head self-attention mechanism"
        },
        {
            "query": "dropout regularization",
            "code_snippet": """class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        return x""",
            "paper_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "paper_context": "Dropout randomly drops units during training to prevent co-adaptation"
        },
        {
            "query": "batch normalization",
            "code_snippet": """def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta""",
            "paper_title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
            "paper_context": "Normalizes layer inputs to reduce internal covariate shift"
        }
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}/{len(test_cases)} ---")
        print(f"Query: {test_case['query']}")

        try:
            response = requests.post(
                "http://localhost:8000/explain",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success")
                print(f"Explanation: {result['explanation'][:100]}...")
                results.append(True)
            else:
                print(f"❌ Failed: {response.status_code}")
                results.append(False)

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)

    print(f"\n=== Summary ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    return all(results)


def main():
    """Run all tests"""
    print("="*60)
    print("API Endpoint Testing")
    print("="*60)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set!")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return

    # Check if server is running
    try:
        requests.get("http://localhost:8000/", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ API server not running!")
        print("Please start the server first:")
        print("  python3 -m uvicorn src.api.app:app --reload")
        return

    # Run tests
    health_ok = test_health_check()
    explain_ok = test_explain_endpoint()
    multiple_ok = test_multiple_examples()

    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"  Single Explain: {'✅ PASS' if explain_ok else '❌ FAIL'}")
    print(f"  Multiple Examples: {'✅ PASS' if multiple_ok else '❌ FAIL'}")
    print("="*60)


if __name__ == "__main__":
    main()
