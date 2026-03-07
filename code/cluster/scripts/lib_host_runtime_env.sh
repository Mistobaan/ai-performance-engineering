#!/usr/bin/env bash

# Shared helpers for scripts that execute the host parity venv directly.

host_runtime_env_file() {
  local root_dir="${1:?root_dir is required}"
  printf '%s\n' "${HOST_RUNTIME_ENV_FILE:-${root_dir}/env/venv/orig_parity_runtime_env.sh}"
}

source_host_runtime_env_if_present() {
  local root_dir="${1:?root_dir is required}"
  local env_file
  env_file="$(host_runtime_env_file "$root_dir")"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
}

host_runtime_remote_prefix() {
  local root_dir="${1:?root_dir is required}"
  local env_file
  env_file="$(host_runtime_env_file "$root_dir")"
  printf 'if [[ -f %q ]]; then source %q >/dev/null 2>&1; fi; ' "$env_file" "$env_file"
}
