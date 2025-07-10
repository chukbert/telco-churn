#!/usr/bin/env python3
"""
Test SageMaker Endpoint for Telco Customer Churn Prediction
Provides comprehensive testing of deployed endpoints
"""

import boto3
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndpointTester:
    """Comprehensive testing for SageMaker endpoints"""
    
    def __init__(self, endpoint_name, region='us-east-1'):
        self.endpoint_name = endpoint_name
        self.region = region
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        
    def invoke_endpoint(self, payload, content_type='application/json'):
        """Invoke the SageMaker endpoint"""
        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType=content_type,
                Body=json.dumps(payload) if content_type == 'application/json' else payload
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
            
        except Exception as e:
            logger.error(f"Error invoking endpoint: {e}")
            raise
    
    def test_single_prediction(self):
        """Test with a single customer record"""
        logger.info("Testing single customer prediction...")
        
        # High churn risk customer
        high_risk_customer = {
            "Gender": "Female",
            "Senior Citizen": 1,
            "Partner": "No",
            "Dependents": "No", 
            "Tenure Months": 1,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No", 
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 85.0,
            "Total Charges": 85.0
        }
        
        result = self.invoke_endpoint(high_risk_customer)
        
        logger.info("Single prediction result:")
        logger.info(json.dumps(result, indent=2))
        
        return result
    
    def test_batch_predictions(self):
        """Test with multiple customer records"""
        logger.info("Testing batch predictions...")
        
        # Multiple test customers with different risk profiles
        test_customers = [
            # High risk: Short tenure, month-to-month, fiber optic, no add-ons
            {
                "Gender": "Male",
                "Senior Citizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "Tenure Months": 2,
                "Phone Service": "Yes", 
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes", 
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 89.9,
                "Total Charges": 179.8
            },
            # Low risk: Long tenure, two-year contract, multiple add-ons
            {
                "Gender": "Female",
                "Senior Citizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "Tenure Months": 65,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes", 
                "Internet Service": "DSL",
                "Online Security": "Yes",
                "Online Backup": "Yes",
                "Device Protection": "Yes",
                "Tech Support": "Yes",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Two year",
                "Paperless Billing": "No",
                "Payment Method": "Bank transfer (automatic)",
                "Monthly Charges": 75.2,
                "Total Charges": 4888.8
            },
            # Medium risk: Medium tenure, one-year contract
            {
                "Gender": "Male",
                "Senior Citizen": 1,
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 24,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes",
                "Internet Service": "Fiber optic",
                "Online Security": "Yes",
                "Online Backup": "No",
                "Device Protection": "Yes",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "No",
                "Contract": "One year",
                "Paperless Billing": "Yes", 
                "Payment Method": "Credit card (automatic)",
                "Monthly Charges": 80.15,
                "Total Charges": 1923.6
            }
        ]
        
        results = []
        for i, customer in enumerate(test_customers):
            logger.info(f"Testing customer {i+1}/3...")
            result = self.invoke_endpoint(customer)
            results.append({
                'customer_id': i+1,
                'input': customer,
                'prediction': result
            })
        
        logger.info("Batch prediction results:")
        for result in results:
            pred = result['prediction'][0] if isinstance(result['prediction'], list) else result['prediction']
            logger.info(f"Customer {result['customer_id']}: "
                       f"Churn Probability = {pred['churn_probability']:.3f}, "
                       f"Prediction = {pred['churn_label']}, "
                       f"Confidence = {pred['confidence']}")
        
        return results
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        logger.info("Testing edge cases...")
        
        test_cases = []
        
        # Test 1: Missing some fields
        try:
            incomplete_customer = {
                "Gender": "Female",
                "Tenure Months": 12,
                "Monthly Charges": 50.0
                # Missing many fields
            }
            result = self.invoke_endpoint(incomplete_customer)
            test_cases.append({"case": "Missing fields", "success": True, "result": result})
        except Exception as e:
            test_cases.append({"case": "Missing fields", "success": False, "error": str(e)})
        
        # Test 2: Invalid data types
        try:
            invalid_customer = {
                "Gender": "Female",
                "Senior Citizen": "invalid",  # Should be 0 or 1
                "Tenure Months": "not_a_number",  # Should be numeric
                "Monthly Charges": 50.0,
                "Total Charges": 600.0
            }
            result = self.invoke_endpoint(invalid_customer)
            test_cases.append({"case": "Invalid data types", "success": True, "result": result})
        except Exception as e:
            test_cases.append({"case": "Invalid data types", "success": False, "error": str(e)})
        
        # Test 3: Empty payload
        try:
            result = self.invoke_endpoint({})
            test_cases.append({"case": "Empty payload", "success": True, "result": result})
        except Exception as e:
            test_cases.append({"case": "Empty payload", "success": False, "error": str(e)})
        
        logger.info("Edge case test results:")
        for case in test_cases:
            if case["success"]:
                logger.info(f"✅ {case['case']}: Handled successfully")
            else:
                logger.info(f"❌ {case['case']}: {case['error']}")
        
        return test_cases
    
    def performance_test(self, num_requests=10):
        """Test endpoint performance"""
        logger.info(f"Running performance test with {num_requests} requests...")
        
        # Standard test customer
        test_customer = {
            "Gender": "Female",
            "Senior Citizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure Months": 12,
            "Phone Service": "Yes",
            "Multiple Lines": "No", 
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "Yes",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "One year",
            "Paperless Billing": "No",
            "Payment Method": "Bank transfer (automatic)",
            "Monthly Charges": 45.2,
            "Total Charges": 542.4
        }
        
        response_times = []
        
        for i in range(num_requests):
            start_time = datetime.now()
            try:
                result = self.invoke_endpoint(test_customer)
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000  # ms
                response_times.append(response_time)
                logger.info(f"Request {i+1}/{num_requests}: {response_time:.2f}ms")
            except Exception as e:
                logger.error(f"Request {i+1} failed: {e}")
        
        if response_times:
            avg_response_time = np.mean(response_times)
            min_response_time = np.min(response_times)
            max_response_time = np.max(response_times)
            p95_response_time = np.percentile(response_times, 95)
            
            logger.info(f"Performance Test Results:")
            logger.info(f"  Average response time: {avg_response_time:.2f}ms")
            logger.info(f"  Min response time: {min_response_time:.2f}ms")
            logger.info(f"  Max response time: {max_response_time:.2f}ms")
            logger.info(f"  95th percentile: {p95_response_time:.2f}ms")
            logger.info(f"  Success rate: {len(response_times)}/{num_requests} ({len(response_times)/num_requests*100:.1f}%)")
            
            return {
                "avg_response_time_ms": avg_response_time,
                "min_response_time_ms": min_response_time,
                "max_response_time_ms": max_response_time,
                "p95_response_time_ms": p95_response_time,
                "success_rate": len(response_times) / num_requests,
                "total_requests": num_requests
            }
        else:
            logger.error("All requests failed!")
            return None
    
    def comprehensive_test(self):
        """Run all tests and generate a comprehensive report"""
        logger.info(f"Starting comprehensive test of endpoint: {self.endpoint_name}")
        
        report = {
            "endpoint_name": self.endpoint_name,
            "region": self.region,
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        try:
            # Test 1: Single prediction
            logger.info("\n" + "="*50)
            logger.info("TEST 1: Single Prediction")
            logger.info("="*50)
            report["tests"]["single_prediction"] = self.test_single_prediction()
            
            # Test 2: Batch predictions
            logger.info("\n" + "="*50)
            logger.info("TEST 2: Batch Predictions")
            logger.info("="*50)
            report["tests"]["batch_predictions"] = self.test_batch_predictions()
            
            # Test 3: Edge cases
            logger.info("\n" + "="*50)
            logger.info("TEST 3: Edge Cases")
            logger.info("="*50)
            report["tests"]["edge_cases"] = self.test_edge_cases()
            
            # Test 4: Performance
            logger.info("\n" + "="*50)
            logger.info("TEST 4: Performance Test")
            logger.info("="*50)
            report["tests"]["performance"] = self.performance_test()
            
            # Overall assessment
            logger.info("\n" + "="*50)
            logger.info("OVERALL ASSESSMENT")
            logger.info("="*50)
            
            # Check if basic functionality works
            single_test_success = "churn_probability" in str(report["tests"]["single_prediction"])
            batch_test_success = len(report["tests"]["batch_predictions"]) > 0
            performance_success = report["tests"]["performance"] is not None
            
            overall_status = "PASS" if all([single_test_success, batch_test_success, performance_success]) else "PARTIAL"
            
            logger.info(f"Endpoint Status: {overall_status}")
            logger.info(f"✅ Single predictions: {'Working' if single_test_success else 'Failed'}")
            logger.info(f"✅ Batch predictions: {'Working' if batch_test_success else 'Failed'}")
            logger.info(f"✅ Performance test: {'Working' if performance_success else 'Failed'}")
            
            if performance_success:
                avg_time = report["tests"]["performance"]["avg_response_time_ms"]
                logger.info(f"📊 Average response time: {avg_time:.2f}ms")
            
            report["overall_status"] = overall_status
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            report["overall_status"] = "FAILED"
            report["error"] = str(e)
        
        # Save report
        report_filename = f"endpoint_test_report_{self.endpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\nTest report saved to: {report_filename}")
        return report

def main():
    parser = argparse.ArgumentParser(description='Test SageMaker Endpoint')
    
    parser.add_argument('--endpoint-name', type=str, required=True,
                       help='SageMaker endpoint name to test')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    parser.add_argument('--test-type', type=str, default='comprehensive',
                       choices=['single', 'batch', 'edge', 'performance', 'comprehensive'],
                       help='Type of test to run')
    parser.add_argument('--num-requests', type=int, default=10,
                       help='Number of requests for performance test')
    
    args = parser.parse_args()
    
    try:
        tester = EndpointTester(args.endpoint_name, args.region)
        
        if args.test_type == 'single':
            result = tester.test_single_prediction()
        elif args.test_type == 'batch':
            result = tester.test_batch_predictions()
        elif args.test_type == 'edge':
            result = tester.test_edge_cases()
        elif args.test_type == 'performance':
            result = tester.performance_test(args.num_requests)
        else:  # comprehensive
            result = tester.comprehensive_test()
        
        logger.info("Testing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())