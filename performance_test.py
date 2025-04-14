import time
import requests
from concurrent.futures import ThreadPoolExecutor

def test_api_performance():
    def make_request():
        try:
            response = requests.get('http://localhost:8000/cctv/streams')
            return response.status_code
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda x: make_request(), range(100)))
    
    end_time = time.time()
    
    successful_requests = len([r for r in results if r == 200])
    total_time = end_time - start_time
    
    print(f"Total requests: 100")
    print(f"Successful requests: {successful_requests}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {100/total_time:.2f}")

if __name__ == "__main__":
    test_api_performance()
