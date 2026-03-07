terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Latest Amazon Linux 2023 AMI
# ---------------------------------------------------------------------------
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------------
# Security Group — Allow SSH (22) + Streamlit (8501)
# ---------------------------------------------------------------------------
resource "aws_security_group" "streamlit_sg" {
  name        = "streamlit-stock-ai-sg"
  description = "Allow SSH and Streamlit traffic"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Streamlit default port
  ingress {
    description = "Streamlit"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "streamlit-stock-ai-sg"
    Project = "stock-ai"
  }
}

# ---------------------------------------------------------------------------
# EC2 Instance — t2.micro (free tier)
# ---------------------------------------------------------------------------
resource "aws_instance" "streamlit_app" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = var.key_name != "" ? var.key_name : null
  vpc_security_group_ids = [aws_security_group.streamlit_sg.id]

  # 8 GB root volume (free tier allows up to 30 GB)
  root_block_device {
    volume_size = 8
    volume_type = "gp3"
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    mistral_api_key  = var.mistral_api_key
    newsdata_api_key = var.newsdata_api_key
    hf_api_key       = var.hf_api_key
  }))

  tags = {
    Name    = "stock-ai-streamlit"
    Project = "stock-ai"
  }
}
