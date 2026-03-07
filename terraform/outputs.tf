output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.streamlit_app.id
}

output "public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.streamlit_app.public_ip
}

output "app_url" {
  description = "URL to access the Streamlit app"
  value       = "http://${aws_instance.streamlit_app.public_ip}:8501"
}
