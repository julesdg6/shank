#!/bin/sh
set -eu

exec supervisord -c /app/docker/supervisord.conf
