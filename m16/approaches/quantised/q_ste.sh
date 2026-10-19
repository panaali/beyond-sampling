# T352 amendment 1: binary cells with fallback STEs (two runs in parallel), then float64 1e8 evaluation
cd /content/cn/m16/approaches/quantised
( python qtrain.py qat_ht3 --ste ht3 --spec bin_s0:0:1:0 bin_s1:1:1:0 bin_s2:2:1:0 bin_s3:3:1:0 bin_s4:4:1:0 bin_x16_s0:0:1:16 bin_x256_s0:0:1:256 bin_x16_s1:1:1:16 bin_x256_s1:1:1:256 bin_x16_s2:2:1:16 bin_x256_s2:2:1:256  > /content/qat_ht3.log 2>&1 && python qeval.py qat_ht3.pt 1e8 > /content/qat_ht3_eval.log 2>&1; echo "rc=$?" >> /content/qat_ht3_eval.log ) &
( python qtrain.py qat_tanh --ste tanh --spec bin_s0:0:1:0 bin_s1:1:1:0 bin_s2:2:1:0 bin_s3:3:1:0 bin_s4:4:1:0 bin_x16_s0:0:1:16 bin_x256_s0:0:1:256 bin_x16_s1:1:1:16 bin_x256_s1:1:1:256 bin_x16_s2:2:1:16 bin_x256_s2:2:1:256  > /content/qat_tanh.log 2>&1 && python qeval.py qat_tanh.pt 1e8 > /content/qat_tanh_eval.log 2>&1; echo "rc=$?" >> /content/qat_tanh_eval.log ) &
wait
