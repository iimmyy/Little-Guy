#!/bin/bash
# The compile VM's nightly check-up of the oven, run at every boot by lg-nightly.service. The VM is woken at 00:30
# UTC by the lg-wake job and shut down at 07:00 UTC by its auto-shutdown schedule.
#   1. fetch the engine source snapshot from storage (tools/azure/push_source.sh puts it there)
#   2. build it for x86 with tests and run the CPU op tests (catches a broken recipe before it reaches the board)
#   3. cross-build the board's CPU oven (aarch64, dotprod) and keep it in storage, dated
# Signs in with the VM's own managed identity; no passwords or keys live on the machine.
set -uo pipefail
ST=$(cat /opt/lg/storage-account)
DAY=$(date -u +%F)
W=/opt/lg/work
LOG=/opt/lg/nightly-$DAY.log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date -u) nightly start"

until az login --identity -o none 2>/dev/null; do sleep 15; done   # the identity endpoint needs a moment after boot
blob() { az storage blob "$@" --account-name "$ST" --auth-mode login --no-progress -o none; }

rm -rf "$W" && mkdir -p "$W" && cd "$W"
blob download -c build -n lcpp-src.tar.gz -f src.tar.gz || { echo "no source snapshot in storage"; exit 1; }
mkdir src && tar xzf src.tar.gz -C src
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF)
# a source snapshot has no git history, so the server's web-page bundle can't find its version; pin the one the
# board's build uses (the bundle is only the browser UI, unused by Little Guy)
export HF_UI_VERSION=b9784

status=ok
cmake -S src -B x86 "${COMMON[@]}" -DLLAMA_BUILD_TESTS=ON && cmake --build x86 -j "$(nproc)" || status="x86 build failed"
if [ "$status" = ok ]; then
    x86/bin/test-backend-ops -b CPU > test-backend-ops.txt 2>&1 || status="CPU op tests failed"
    tail -3 test-backend-ops.txt
fi

# a portable x86 server for the nightly brain tests (any AVX2 machine; no shared libraries to carry around)
cmake -S src -B x86-portable "${COMMON[@]}" -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    && cmake --build x86-portable -j "$(nproc)" --target llama-server \
    && blob upload -c build -n nightly/latest/llama-server-x86 -f x86-portable/bin/llama-server --overwrite \
    || status="${status}; portable x86 server failed"

cmake -S src -B arm "${COMMON[@]}" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF \
    && cmake --build arm -j "$(nproc)" --target llama-server llama-bench \
    && tar czf board-cpu-oven.tar.gz -C arm bin \
    && blob upload -c build -n "nightly/$DAY/board-cpu-oven.tar.gz" -f board-cpu-oven.tar.gz --overwrite \
    || status="${status}; board cross-build failed"

[ -f test-backend-ops.txt ] && blob upload -c results -n "nightly/$DAY/test-backend-ops.txt" -f test-backend-ops.txt --overwrite
echo "== $(date -u) nightly done: $status"
blob upload -c results -n "nightly/$DAY/nightly.log" -f "$LOG" --overwrite
