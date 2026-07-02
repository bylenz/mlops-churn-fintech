output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.this.repository_url
}

output "api_url" {
  description = "Public API URL via ALB"
  value       = "http://${aws_lb.this.dns_name}"
}

output "grafana_url" {
  description = "Grafana URL via ALB"
  value       = "http://${aws_lb.this.dns_name}:3000"
}

output "ecs_cluster" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service" {
  value = aws_ecs_service.api.name
}
