#!/usr/bin/env pwsh
# Quickstart — compile the notes into adapters and inspect them.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

llm-patch compile ./notes --output ./out
llm-patch adapter status ./out
