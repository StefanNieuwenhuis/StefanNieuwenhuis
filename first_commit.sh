#!/bin/bash

# Initialize Git repo if not already initialized
if [ ! -d ".git" ]; then
  git init
  echo "Initialized a new Git repository."
fi

# Create the basic folder structure
mkdir -p _posts/dsa _posts/recommender-system _posts/homelab
mkdir -p assets images _includes _layouts

# Add Minimal Mistakes remote theme to Gemfile
cat <<EOL > Gemfile
source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins
gem "jekyll-remote-theme"
EOL

# Create a config file for Jekyll
cat <<EOL > _config.yml
title: "Stefan Nieuwenhuis"
description: "Learning in public — ML, recommender systems, homelabs, and more."
baseurl: ""
url: "https://stefannieuwenhuis.github.io"
theme: minimal-mistakes-jekyll
remote_theme: mmistakes/minimal-mistakes

plugins:
  - jekyll-feed
  - jekyll-remote-theme
  - jekyll-seo-tag
  - jekyll-paginate

collections:
  posts:
    output: true

defaults:
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true

markdown: kramdown
kramdown:
  input: GFM
  math_engine: mathjax

include: ["_pages"]
EOL

# Create a basic README
cat <<EOL > README.md
# Stefan Nieuwenhuis Blog

This is my GitHub Pages blog built with [Jekyll](https://jekyllrb.com/) and the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme.

## Topics I blog about

- 🧠 Data Structures & Algorithms
- 🤖 Recommender Systems
- 🏠 Homelab & Kubernetes on bare metal

Built with learning, reflection, and visibility in mind.
EOL

# Add .gitignore
cat <<EOL > .gitignore
_site/
.sass-cache/
.jekyll-cache/
*.gem
*.rbc
.bundle/
vendor/
EOL

# # Add everything and commit
# git add .
# git commit -m "Initial commit: Setup Jekyll blog with Minimal Mistakes theme and post structure"

# # Push to GitHub (main branch)
# git branch -M main
# git push -u origin main
