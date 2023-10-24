---
layout: page
title: About
permalink: /about/
---

### Quicksart Guide

- Install Prerequisites

```shell
ruby -v
gem -v
gcc -v
g++ -v
make -v
```

- Install Ruby

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install chruby ruby-install xz

ruby-install ruby

echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.bash_profile
echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.bash_profile
echo "chruby ruby-3.1.2" >> ~/.bash_profile

ruby -v
```

- Install Jekyll and Bundler from gem

```shell
gem update --system
gem source -a http://rubygems.org
gem source --remove https://rubygems.org
gem install jekyll bundler
```

- Start a new blog project [Optional]

```shell

jekyll new docs
cd docs
bundle exec jekyll serve
# bundle info --path minima
```

- Build with incremential option

```shell
jekyll build --incremental
```

- Serve with auto-reload option

```shell
bundle exec jekyll serve --livereload
```
