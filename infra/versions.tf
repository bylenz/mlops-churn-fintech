terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # For teams, uncomment and configure an S3 backend for shared state.
  # backend "s3" {
  #   bucket = "your-tf-state-bucket"
  #   key    = "churn-predictor/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}
