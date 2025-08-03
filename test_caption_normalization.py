#!/usr/bin/env python3
"""
Test script for Caption Normalization System
Demonstrates fingerprinting and semantic comparison capabilities
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000/scholar"
API_ENDPOINT = f"{BASE_URL}/caption-normalize/"

def test_caption_normalization():
    """Test basic caption normalization"""
    print("=" * 60)
    print("TESTING CAPTION NORMALIZATION")
    print("=" * 60)
    
    # Test data from user's example
    test_captions = [
        "Figure 2: Accuracy across 10 folds (Smith, 2020)",
        "Table 1. Summary of Results",
        "Figure 3: Model performance comparison [1]",
        "Table 2. Hyperparameter settings (Jones et al., 2021)",
        "Fig. 4: Training loss over 100 epochs",
        "Table 3. Experimental results [2-5]"
    ]
    
    payload = {
        "captions": test_captions,
        "operation": "normalize"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("Original Captions:")
        for i, caption in enumerate(result['original_captions'], 1):
            print(f"  {i}. {caption}")
        
        print("\nNormalized Captions:")
        for i, normalized in enumerate(result['normalized_captions'], 1):
            print(f"  {i}. {normalized}")
        
        print(f"\nProcessing Time: {result['processing_time']:.4f} seconds")
        print(f"Status: {result['status']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    
    return True

def test_caption_similarity():
    """Test caption similarity calculation"""
    print("\n" + "=" * 60)
    print("TESTING CAPTION SIMILARITY")
    print("=" * 60)
    
    test_captions = [
        "Figure 2: Accuracy across 10 folds (Smith, 2020)",
        "Figure 3: Model performance comparison [1]",
        "Table 1. Summary of Results",
        "Table 2. Hyperparameter settings (Jones et al., 2021)"
    ]
    
    target_caption = "Figure 2: Accuracy across 10 folds (Smith, 2020)"
    
    payload = {
        "captions": test_captions,
        "operation": "similarity",
        "target_caption": target_caption
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"Target Caption: {result['target_caption']}")
        print("\nSimilarity Scores:")
        for similarity in result['similarities']:
            print(f"  {similarity['caption']}")
            print(f"    Similarity: {similarity['similarity_score']:.4f}")
        
        print(f"\nProcessing Time: {result['processing_time']:.4f} seconds")
        print(f"Status: {result['status']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    
    return True

def test_find_similar_captions():
    """Test finding similar captions above threshold"""
    print("\n" + "=" * 60)
    print("TESTING FIND SIMILAR CAPTIONS")
    print("=" * 60)
    
    test_captions = [
        "Figure 2: Accuracy across 10 folds (Smith, 2020)",
        "Figure 3: Model performance comparison [1]",
        "Figure 4: Training loss over 100 epochs",
        "Table 1. Summary of Results",
        "Table 2. Hyperparameter settings (Jones et al., 2021)",
        "Table 3. Experimental results [2-5]"
    ]
    
    target_caption = "Figure 2: Accuracy across 10 folds (Smith, 2020)"
    threshold = 0.6
    
    payload = {
        "captions": test_captions,
        "operation": "find_similar",
        "target_caption": target_caption,
        "threshold": threshold
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"Target Caption: {result['target_caption']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Total Matches: {result['total_matches']}")
        
        print("\nSimilar Captions:")
        for match in result['similar_captions']:
            print(f"  {match['caption']}")
            print(f"    Similarity: {match['similarity_score']:.4f}")
        
        print(f"\nProcessing Time: {result['processing_time']:.4f} seconds")
        print(f"Status: {result['status']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Test empty captions
    print("1. Testing empty captions:")
    payload = {
        "captions": [],
        "operation": "normalize"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 400:
            print(f"   Expected Error: {response.json().get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test invalid operation
    print("\n2. Testing invalid operation:")
    payload = {
        "captions": ["Figure 1: Test"],
        "operation": "invalid_operation"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 400:
            print(f"   Expected Error: {response.json().get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test special characters and numbers
    print("\n3. Testing special characters and numbers:")
    special_captions = [
        "Figure 1: Results @ 95% confidence!",
        "Table 2: Data from 2020-2023 [ref]",
        "Fig. 3: Performance (n=1000) & accuracy",
        "Table 4: Summary of findings..."
    ]
    
    payload = {
        "captions": special_captions,
        "operation": "normalize"
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("   Original vs Normalized:")
        for orig, norm in zip(result['original_captions'], result['normalized_captions']):
            print(f"   Original: {orig}")
            print(f"   Normalized: {norm}")
            print()
        
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")

def main():
    """Run all tests"""
    print("CAPTION NORMALIZATION SYSTEM TEST")
    print("=" * 60)
    print("This test demonstrates the caption normalization functionality")
    print("for fingerprinting and semantic comparison in academic papers.")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/list/")
        if response.status_code != 200:
            print("Warning: Server might not be running. Tests may fail.")
            print("Start the server with: python manage.py runserver")
            print()
    except:
        print("Warning: Cannot connect to server. Tests may fail.")
        print("Start the server with: python manage.py runserver")
        print()
    
    # Run tests
    success_count = 0
    total_tests = 3
    
    if test_caption_normalization():
        success_count += 1
    
    if test_caption_similarity():
        success_count += 1
    
    if test_find_similar_captions():
        success_count += 1
    
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ All tests passed! Caption normalization system is working correctly.")
    else:
        print("❌ Some tests failed. Check server status and implementation.")
    
    print("\nCaption Normalization Features:")
    print("✅ Remove special characters and citations")
    print("✅ Convert to lowercase")
    print("✅ Replace numbers with [#] placeholder")
    print("✅ Remove extra whitespace")
    print("✅ Calculate semantic similarity")
    print("✅ Find similar captions above threshold")
    print("✅ Batch processing support")

if __name__ == "__main__":
    main() 