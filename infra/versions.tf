terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Shared state so local bootstrap and CI operate on the same infrastructure.
  backend "s3" {
    bucket = "churn-predictor-tfstate-454014151293"
    key    = "churn-predictor/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}
