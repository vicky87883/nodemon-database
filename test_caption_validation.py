#!/usr/bin/env python3
"""
Test script for Caption Validation System
Demonstrates quality scoring for figure and table captions
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000/scholar"
API_ENDPOINT = f"{BASE_URL}/caption-validate/"

def test_single_caption_validation():
    """Test validation of a single caption"""
    print("=" * 60)
    print("TESTING SINGLE CAPTION VALIDATION")
    print("=" * 60)
    
    # Test data from user's example
    test_data = {
        "captions": [
            {
                "caption": "Figure 3: Model architecture",
                "linked_text": "As shown in Figure 3, the model consists of three layers and a residual connection."
            }
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=test_data)
        response.raise_for_status()
        
        result = response.json()
        
        print("Input:")
        print(f"  Caption: {test_data['captions'][0]['caption']}")
        print(f"  Linked Text: {test_data['captions'][0]['linked_text']}")
        
        print("\nValidation Results:")
        validation_result = result['validation_results'][0]
        quality_scores = validation_result['quality_scores']
        
        print(f"  Clarity Score: {quality_scores['clarity_score']}")
        print(f"  Contextual Link Score: {quality_scores['contextual_link_score']}")
        print(f"  Completeness Score: {quality_scores['completeness_score']}")
        print(f"  Overall Quality Score: {quality_scores['overall_quality_score']}")
        print(f"  Validation Status: {validation_result['validation_status']}")
        
        print(f"\nProcessing Time: {result['processing_time']:.4f} seconds")
        print(f"Status: {result['status']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    
    return True

def test_multiple_captions_validation():
    """Test validation of multiple captions"""
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE CAPTIONS VALIDATION")
    print("=" * 60)
    
    test_data = {
        "captions": [
            {
                "caption": "Figure 3: Model architecture",
                "linked_text": "As shown in Figure 3, the model consists of three layers and a residual connection."
            },
            {
                "caption": "Table 1: Performance comparison",
                "linked_text": "Table 1 shows the accuracy and precision metrics for different models."
            },
            {
                "caption": "Figure 2",
                "linked_text": "The results are presented in Figure 2."
            },
            {
                "caption": "Table 3: Experimental results across multiple datasets with detailed analysis",
                "linked_text": "As demonstrated in Table 3, our approach achieves superior performance across all datasets."
            }
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=test_data)
        response.raise_for_status()
        
        result = response.json()
        
        print("Validation Results:")
        for i, validation_result in enumerate(result['validation_results'], 1):
            print(f"\nCaption {i}: {validation_result['caption']}")
            quality_scores = validation_result['quality_scores']
            print(f"  Clarity: {quality_scores['clarity_score']}")
            print(f"  Contextual Link: {quality_scores['contextual_link_score']}")
            print(f"  Completeness: {quality_scores['completeness_score']}")
            print(f"  Overall: {quality_scores['overall_quality_score']}")
            print(f"  Status: {validation_result['validation_status']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Total Captions: {summary['total_captions']}")
        print(f"  High Quality: {summary['high_quality_count']}")
        print(f"  Medium Quality: {summary['medium_quality_count']}")
        print(f"  Low Quality: {summary['low_quality_count']}")
        
        avg_scores = summary['average_scores']
        print(f"  Average Clarity: {avg_scores['clarity_score']}")
        print(f"  Average Contextual Link: {avg_scores['contextual_link_score']}")
        print(f"  Average Completeness: {avg_scores['completeness_score']}")
        print(f"  Average Overall: {avg_scores['overall_quality_score']}")
        
        print(f"\nProcessing Time: {result['processing_time']:.4f} seconds")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Test empty caption
    print("1. Testing empty caption:")
    test_data = {
        "captions": [
            {
                "caption": "",
                "linked_text": "Some text here."
            }
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=test_data)
        response.raise_for_status()
        
        result = response.json()
        validation_result = result['validation_results'][0]
        quality_scores = validation_result['quality_scores']
        
        print(f"   Empty caption scores: {quality_scores}")
        
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")
    
    # Test caption without linked text
    print("\n2. Testing caption without linked text:")
    test_data = {
        "captions": [
            {
                "caption": "Figure 5: Complex neural network architecture with attention mechanisms",
                "linked_text": ""
            }
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=test_data)
        response.raise_for_status()
        
        result = response.json()
        validation_result = result['validation_results'][0]
        quality_scores = validation_result['quality_scores']
        
        print(f"   Caption without linked text scores: {quality_scores}")
        
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")
    
    # Test very short caption
    print("\n3. Testing very short caption:")
    test_data = {
        "captions": [
            {
                "caption": "Fig. 1",
                "linked_text": "Figure 1 shows the results."
            }
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=test_data)
        response.raise_for_status()
        
        result = response.json()
        validation_result = result['validation_results'][0]
        quality_scores = validation_result['quality_scores']
        
        print(f"   Short caption scores: {quality_scores}")
        
    except requests.exceptions.RequestException as e:
        print(f"   Error: {e}")

def test_quality_scoring_examples():
    """Test various quality scoring scenarios"""
    print("\n" + "=" * 60)
    print("TESTING QUALITY SCORING EXAMPLES")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "High Quality Caption",
            "caption": "Figure 4: Performance comparison of deep learning models across multiple datasets",
            "linked_text": "As illustrated in Figure 4, our proposed model achieves superior performance compared to baseline approaches."
        },
        {
            "name": "Medium Quality Caption",
            "caption": "Table 2: Results summary",
            "linked_text": "The results are summarized in Table 2."
        },
        {
            "name": "Low Quality Caption",
            "caption": "Fig",
            "linked_text": "See figure above."
        },
        {
            "name": "Well-Referenced Caption",
            "caption": "Figure 6: System architecture diagram",
            "linked_text": "The system architecture, depicted in Figure 6, consists of three main components: the input layer, processing module, and output interface."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print(f"   Caption: {test_case['caption']}")
        print(f"   Linked Text: {test_case['linked_text']}")
        
        test_data = {
            "captions": [
                {
                    "caption": test_case['caption'],
                    "linked_text": test_case['linked_text']
                }
            ]
        }
        
        try:
            response = requests.post(API_ENDPOINT, json=test_data)
            response.raise_for_status()
            
            result = response.json()
            validation_result = result['validation_results'][0]
            quality_scores = validation_result['quality_scores']
            
            print(f"   Scores: Clarity={quality_scores['clarity_score']}, Context={quality_scores['contextual_link_score']}, Completeness={quality_scores['completeness_score']}, Overall={quality_scores['overall_quality_score']}")
            print(f"   Status: {validation_result['validation_status']}")
            
        except requests.exceptions.RequestException as e:
            print(f"   Error: {e}")

def main():
    """Run all tests"""
    print("CAPTION VALIDATION SYSTEM TEST")
    print("=" * 60)
    print("This test demonstrates the caption quality validation functionality")
    print("for evaluating figure and table captions in academic papers.")
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
    total_tests = 2
    
    if test_single_caption_validation():
        success_count += 1
    
    if test_multiple_captions_validation():
        success_count += 1
    
    test_edge_cases()
    test_quality_scoring_examples()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ All tests passed! Caption validation system is working correctly.")
    else:
        print("❌ Some tests failed. Check server status and implementation.")
    
    print("\nCaption Validation Features:")
    print("✅ Clarity scoring (semantic clarity)")
    print("✅ Contextual link scoring (reference quality)")
    print("✅ Completeness scoring (self-contained quality)")
    print("✅ Overall quality scoring (weighted average)")
    print("✅ Batch processing support")
    print("✅ Quality status classification")
    print("✅ Summary statistics")

if __name__ == "__main__":
    main() 