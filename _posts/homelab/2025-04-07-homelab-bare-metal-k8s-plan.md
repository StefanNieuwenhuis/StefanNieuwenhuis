---
title: "Planning My Bare Metal K8s Homelab: Hardware, Goals & Learning Path"
categories: homelab
tags: [kubernetes, bare-metal, devops, learning-in-public]
layout: single
author_profile: true
---

I’m building a Kubernetes homelab with:

- 3x Lenovo ThinkCentre M70s (i5-10400, 16GB RAM)
- 1x Raspberry Pi 4 (2GB RAM)
- 1x USB 3.0 5TB HDD DRIVE
- 1x UniFi 8-port managed switch

The goal is to replicate production-like environments using real K8s (not K3s), running on bare metal. I’ll use this setup to deploy real projects (like my recommender system), monitor them, and refine my DevOps skillset.

For now, this project is in the planning phase. In the meantime, I’m simulating deployments locally using Docker Compose to avoid waiting for hardware availability.

This blog series will cover every step — from cluster planning to wiring, provisioning, setup automation, and real workloads.
