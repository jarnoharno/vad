#!/bin/bash
# $1 : sound file list
# $2 : vad output directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
matlab -nodisplay -r "try;path('${DIR}',path);g729('${1}','${2}');catch e;fprintf('%s\n',e.message);end;exit;" | tail -n +11
