;; Potentia field sum, WebAssembly SIMD (2 adjacent probes per iteration).
;; For every probe (pi, pj):  ex = sum_p kx[p] * (V[base - lin1[p]] - V[base - lin2[p]]),  ey likewise,  E = sqrt(ex*ex + ey*ey + eps)
;; Same IEEE operations in the same order as the JavaScript pair loop in core.js, so the results are bit-identical to it.
;; lin1/lin2 hold byte offsets; V is the zero-padded source array, E the probe grid (row length L1); memory is imported so JS sizes it.
(module
  (import "env" "mem" (memory 1))
  (func (export "field")
    (param $vp i32) (param $e i32) (param $l1 i32) (param $l2 i32) (param $kx i32) (param $ky i32)
    (param $np i32) (param $L0 i32) (param $L1 i32) (param $off i32) (param $pw1 i32) (param $eps f64)
    (local $pi i32) (local $pj i32) (local $row i32) (local $base i32) (local $p i32) (local $addr i32)
    (local $ex v128) (local $ey v128) (local $d v128)
    (block $done_pi (loop $next_pi
      (br_if $done_pi (i32.ge_u (local.get $pi) (local.get $L0)))
      (local.set $row (i32.add (i32.mul (i32.add (local.get $pi) (local.get $off)) (local.get $pw1)) (local.get $off)))
      (local.set $pj (i32.const 0))
      (block $done_pj (loop $next_pj
        (br_if $done_pj (i32.ge_u (local.get $pj) (local.get $L1)))
        (local.set $base (i32.add (local.get $vp) (i32.shl (i32.add (local.get $row) (local.get $pj)) (i32.const 3))))
        (local.set $ex (v128.const f64x2 0 0))
        (local.set $ey (v128.const f64x2 0 0))
        (local.set $p (i32.const 0))
        (block $done_p (loop $next_p
          (br_if $done_p (i32.ge_u (local.get $p) (local.get $np)))
          (local.set $d (f64x2.sub
            (v128.load (i32.sub (local.get $base) (i32.load (i32.add (local.get $l1) (i32.shl (local.get $p) (i32.const 2))))))
            (v128.load (i32.sub (local.get $base) (i32.load (i32.add (local.get $l2) (i32.shl (local.get $p) (i32.const 2))))))))
          (local.set $ex (f64x2.add (local.get $ex)
            (f64x2.mul (v128.load64_splat (i32.add (local.get $kx) (i32.shl (local.get $p) (i32.const 3)))) (local.get $d))))
          (local.set $ey (f64x2.add (local.get $ey)
            (f64x2.mul (v128.load64_splat (i32.add (local.get $ky) (i32.shl (local.get $p) (i32.const 3)))) (local.get $d))))
          (local.set $p (i32.add (local.get $p) (i32.const 1)))
          (br $next_p)))
        (local.set $d (f64x2.sqrt (f64x2.add
          (f64x2.add (f64x2.mul (local.get $ex) (local.get $ex)) (f64x2.mul (local.get $ey) (local.get $ey)))
          (f64x2.splat (local.get $eps)))))
        (local.set $addr (i32.add (local.get $e) (i32.shl (i32.add (i32.mul (local.get $pi) (local.get $L1)) (local.get $pj)) (i32.const 3))))
        (f64.store (local.get $addr) (f64x2.extract_lane 0 (local.get $d)))
        (if (i32.lt_u (i32.add (local.get $pj) (i32.const 1)) (local.get $L1))
          (then (f64.store offset=8 (local.get $addr) (f64x2.extract_lane 1 (local.get $d)))))
        (local.set $pj (i32.add (local.get $pj) (i32.const 2)))
        (br $next_pj)))
      (local.set $pi (i32.add (local.get $pi) (i32.const 1)))
      (br $next_pi)))))
