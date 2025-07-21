
# $1 indicates the subdirectory (i.e. models or data)
# $2 indicates the benchmark name (i.e. ResNet50, dlrm, etc.)
# $3 indicates the destination filename
function download_sdxl_file {
    _SUB_DIR=$1/$2

    if [ ! -d ${_SUB_DIR} ]; then
        echo "Creating directory ${_SUB_DIR}"
        mkdir -p ${_SUB_DIR}
    fi
    echo "Downloading $2 $1..." \
        && echo ${_SUB_DIR}/$3 \
        && sudo -v ; curl https://rclone.org/install.sh | sudo bash \
        && rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
    cd ${SUB_DIR} 
    sudo rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp16 ${_SUB_DIR} -P --no-check-certificate
    echo "Saved $2 $1 to ${_SUB_DIR}/$3!"
    cd -
}
