variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project / resource name prefix"
  type        = string
  default     = "churn-predictor-prod"
}

variable "image_tag" {
  description = "ECR image tag for the API task definition"
  type        = string
  default     = "latest"
}

variable "grafana_admin_password" {
  description = "Grafana admin password (min 8 chars)"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.grafana_admin_password) >= 8
    error_message = "grafana_admin_password must be at least 8 characters."
  }
}

variable "api_cpu" {
  type    = number
  default = 512
}

variable "api_memory" {
  type    = number
  default = 1024
}
