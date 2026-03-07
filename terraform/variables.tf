variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (t2.micro = free tier)"
  type        = string
  default     = "t2.micro"
}

variable "key_name" {
  description = "Name of an existing EC2 Key Pair for SSH access (optional, leave empty to skip SSH)"
  type        = string
  default     = ""
}

# ----- API Keys (passed securely via tfvars) -----

variable "mistral_api_key" {
  description = "Mistral AI API key"
  type        = string
  sensitive   = true
}

variable "newsdata_api_key" {
  description = "NEWSDATA.io API key"
  type        = string
  sensitive   = true
}

variable "hf_api_key" {
  description = "HuggingFace API key for BERT embeddings"
  type        = string
  sensitive   = true
}
