#!/usr/bin/env python3
"""
Get HuggingFace TEI Container Image URI
This script just gets the URI without deploying anything
"""

import boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri


def get_tei_image_uri(instance_type="ml.g5.xlarge", region="eu-west-1"):
    """
    Get the correct TEI image URI for your instance type and region
    """
    print(f"🔍 Getting TEI image URI for:")
    print(f"   Instance Type: {instance_type}")
    print(f"   Region: {region}")

    # Determine if GPU or CPU based on instance type
    is_gpu = any(gpu_type in instance_type for gpu_type in ['g5', 'p3', 'p4'])
    container_type = "GPU" if is_gpu else "CPU"

    print(f"   Container Type: {container_type}")

    try:
        # Use the official SageMaker method
        key = "huggingface-tei" if is_gpu else "huggingface-tei-cpu"
        image_uri = get_huggingface_llm_image_uri(key, version="1.2.3", region=region)

        print(f"\n✅ TEI Image URI (Official Method):")
        print(f"   {image_uri}")

        return image_uri

    except Exception as e:
        print(f"\n❌ Official method failed: {e}")

        # Fallback: construct manually
        print(f"🔄 Trying fallback method...")
        fallback_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-tei:1.2.3-{'gpu' if is_gpu else 'cpu'}"

        print(f"\n💡 Fallback Image URI:")
        print(f"   {fallback_uri}")

        return fallback_uri


def check_tei_availability(regions=None, instance_types=None):
    """
    Check TEI availability across multiple regions and instance types
    """
    if regions is None:
        regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'eu-central-1', 'ap-southeast-1']

    if instance_types is None:
        instance_types = ['ml.g5.xlarge', 'ml.c6i.2xlarge', 'ml.m5.xlarge']

    print(f"🔍 Checking TEI availability across {len(regions)} regions and {len(instance_types)} instance types...")
    print(f"=" * 80)

    results = {}

    for region in regions:
        print(f"\n📍 Region: {region}")
        results[region] = {}

        for instance_type in instance_types:
            try:
                uri = get_tei_image_uri(instance_type, region)
                results[region][instance_type] = {
                    'status': 'available',
                    'uri': uri
                }
                print(f"   ✅ {instance_type}: Available")

            except Exception as e:
                results[region][instance_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"   ❌ {instance_type}: Failed - {str(e)[:50]}...")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"📊 SUMMARY:")

    for region, instance_results in results.items():
        available_count = sum(1 for r in instance_results.values() if r['status'] == 'available')
        total_count = len(instance_results)
        print(f"   {region}: {available_count}/{total_count} instance types available")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Get HuggingFace TEI Image URI')
    parser.add_argument('--instance-type', default='ml.g5.xlarge', help='SageMaker instance type')
    parser.add_argument('--region', default='eu-west-1', help='AWS region')
    parser.add_argument('--check-all', action='store_true', help='Check availability across multiple regions')

    args = parser.parse_args()

    if args.check_all:
        # Check availability across multiple regions
        results = check_tei_availability()

        # Find the best options
        print(f"\n🎯 RECOMMENDATIONS:")

        best_regions = []
        for region, instance_results in results.items():
            available_count = sum(1 for r in instance_results.values() if r['status'] == 'available')
            if available_count > 0:
                best_regions.append((region, available_count))

        best_regions.sort(key=lambda x: x[1], reverse=True)

        if best_regions:
            for region, count in best_regions[:3]:  # Top 3
                print(f"   ✅ {region} ({count} instance types available)")
        else:
            print(f"   ❌ No regions have TEI containers available")
            print(f"   💡 Try using the original HuggingFace transformers container instead")

    else:
        # Get URI for specific instance type and region
        uri = get_tei_image_uri(args.instance_type, args.region)

        print(f"\n📋 FOR YOUR CLOUDFORMATION TEMPLATE:")
        print(f"   Image: {uri}")

        print(f"\n📋 FOR YOUR PYTHON CODE:")
        print(f"   image_uri = '{uri}'")


if __name__ == "__main__":
    main()