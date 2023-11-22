#!/bin/bash
#
# Run configure for runtime and utils

basedir=$PWD
status=0
fcount=0
runtime_result=""
util_result=""
thedate=$(date)

cd build/darshan-runtime
../../darshan-runtime/configure --enable-apmpi-mod --prefix=$basedir/install --with-jobid-env=DARSHAN_JOBID --with-log-path=$basedir/logs CC=mpicc
runtime_status=$?
if [ $runtime_status -ne 0 ]; then
  fcount=$((fcount+1));
  runtime_result="<error type='$runtime_status' message='configure failed' />"
fi

cd ../darshan-util
../../darshan-util/configure --enable-apmpi-mod --enable-apxc-mod --prefix=$basedir/install
util_status=$?
if [ $util_status -ne 0 ]; then
  fcount=$((fcount+1));
  util_result="<error type='$util_status' message='configure failed' />"
fi

cd ../../;

echo "
<testsuites>
  <testsuite name='configure' tests='2' failures='$fcount' time='$thedate'>
    <testcase name='darshan-runtime' time='$thedate'>
    $runtime_result
    </testcase>
    <testcase name='darshan-util' time='$thedate'>
    $util_result
    </testcase>
  </testsuite>
</testsuites>
" > configure-result.xml

return $fcount
