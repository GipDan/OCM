"""
Streamlit 全生命周期管理：录入、自动训练、导出 CSV、模型覆盖、推理试算。
运行：在项目根目录执行  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocm.database import (  # noqa: E402
    DEFAULT_DB_PATH,
    delete_param_template,
    export_filename_suffix,
    export_records_flat_csv_rows,
    get_connection,
    get_param_template_by_name,
    init_db,
    list_models_for_op_device,
    list_op_device_pairs,
    list_record_export_keys,
    list_param_templates,
    save_param_template,
    upsert_model,
)
from ocm.inference import predict_latency, predict_with_booster_json  # noqa: E402
from ocm.train import fit_and_store_model  # noqa: E402
from ocm.workflow import add_record_maybe_autofit  # noqa: E402

_DEFAULT_PARAMS_JSON = '{"N":1,"C":64,"H":32,"W":32,"is_contiguous":true,"memory_stride":[64,1]}'


def main() -> None:
    st.set_page_config(page_title="OCM 算子代价模型", layout="wide")
    st.title("硬件感知算子代价模型 (OCM)")

    with st.sidebar:
        st.subheader("数据库")
        db_path = st.text_input("SQLite 路径", value=str(DEFAULT_DB_PATH))
        conn = get_connection(db_path)
        init_db(conn)
        st.caption("零文件架构：模型以 JSON 文本存于 `models.model_payload`。")
        n_tpl = len(list_param_templates(conn))
        st.caption(f"已保存 params 模板数：{n_tpl}（与当前库文件绑定）")

    if "record_params" not in st.session_state:
        st.session_state.record_params = _DEFAULT_PARAMS_JSON
    if "infer_params" not in st.session_state:
        st.session_state.infer_params = _DEFAULT_PARAMS_JSON

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["录入数据", "手动训练", "导出 CSV", "模型干预", "推理试算"]
    )

    with tab1:
        st.markdown("录入 benchmark 样本。请选择：**仅写入数据库**，或 **写入后自动训练**（同 `(op_name, device)` 下样本数 ≥ 2 时拟合 XGBoost）。")
        mode = st.radio(
            "录入模式",
            (
                "仅写入 records（不触发训练）",
                "写入 records 后自动训练（样本数 ≥ 2 时拟合）",
            ),
            horizontal=True,
            index=0,
            help="仅写入适合先批量攒数据；自动训练会在每次提交后尝试用当前该算子+设备下的全部样本更新模型。",
        )
        auto_fit = mode.startswith("写入 records 后")

        with st.expander("params 模板（可选）：从库中加载或保存当前编辑区内容", expanded=False):
            tpls = list_param_templates(conn)
            names = [t["name"] for t in tpls]
            r1, r2, r3 = st.columns([2, 1, 1])
            with r1:
                pick_r = st.selectbox("选择模板", ["—"] + names, key="tpl_pick_record")
            with r2:
                st.write("")
                if st.button("加载到编辑区", key="btn_tpl_load_record"):
                    if pick_r != "—":
                        t = get_param_template_by_name(conn, pick_r)
                        if t:
                            st.session_state.record_params = json.dumps(
                                t["params"], ensure_ascii=False, separators=(",", ":")
                            )
                            st.rerun()
            with r3:
                st.write("")
            sn1, sn2 = st.columns([3, 1])
            with sn1:
                st.text_input(
                    "将当前 params 保存为模板（填写名称后点右侧按钮）",
                    key="tpl_save_name_record",
                    placeholder="例如 conv3x3_fp32_baseline",
                )
            with sn2:
                st.write("")
                st.write("")
                if st.button("保存为模板", key="btn_tpl_save_record"):
                    try:
                        p = json.loads(st.session_state.record_params)
                        if not isinstance(p, dict):
                            raise ValueError("params 必须是 JSON 对象")
                        nm = (st.session_state.get("tpl_save_name_record") or "").strip()
                        if not nm:
                            raise ValueError("请填写模板名称")
                        save_param_template(conn, nm, p)
                        st.success(f"已保存模板「{nm}」")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            if names:
                d1, d2 = st.columns([2, 1])
                with d1:
                    del_r = st.selectbox("删除模板", ["—"] + names, key="tpl_del_pick_record")
                with d2:
                    st.write("")
                    if st.button("删除所选", key="btn_tpl_del_record"):
                        if del_r != "—":
                            delete_param_template(conn, del_r)
                            st.success(f"已删除「{del_r}」")
                            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            op_name = st.text_input("op_name", value="nn::conv2d_nchw_fp32")
            device = st.text_input("device", value="NVIDIA_RTX_4090")
            rec_fok = st.text_input(
                "feature_order_key（可选）",
                value="",
                key="rec_fok",
                help="同一算子、设备下区分不同特征模式/样本批次；与导出、训练筛选、模型主键一致。留空表示未标注。",
            )
        with c2:
            st.text_area(
                "params (JSON)",
                height=160,
                key="record_params",
            )
            latency = st.number_input("latency (ms)", min_value=0.0, value=1.452, format="%.6f")
        if st.button("提交"):
            try:
                params = json.loads(st.session_state.record_params)
                if not isinstance(params, dict):
                    raise ValueError("params 必须是 JSON 对象")
                fok = rec_fok.strip() or None
                rid, fit_res = add_record_maybe_autofit(
                    conn,
                    op_name,
                    device,
                    params,
                    latency,
                    auto_fit=auto_fit,
                    feature_order_key=fok,
                )
                st.success(f"已写入记录 id={rid}")
                if fit_res is not None:
                    ok, msg = fit_res
                    if ok:
                        st.success(msg)
                    else:
                        st.info(msg)
            except Exception as e:
                st.error(str(e))

    with tab2:
        st.markdown(
            "针对已有样本手动触发训练（不新增记录）。可选用 **feature_order_key** 只使用带该标签的样本；留空则使用该算子+设备下全部样本。"
        )
        pairs = list_op_device_pairs(conn)
        if not pairs:
            st.warning("暂无 records，请先在「录入」页添加数据。")
        else:
            labels = [f"{a} @ {b}" for a, b in pairs]
            idx = st.selectbox("选择 (op_name, device)", range(len(labels)), format_func=lambda i: labels[i])
            op_sel, dev_sel = pairs[idx]
            st.text(f"op_name: {op_sel}")
            st.text(f"device: {dev_sel}")
            ek = list_record_export_keys(conn, op_sel, dev_sel)
            opts = [("（不筛选，使用全部样本）", "all")]
            if None in ek:
                opts.append(("仅未标注 (feature_order_key 为空)", "null"))
            for k in ek:
                if k is not None:
                    short = k if len(k) <= 72 else k[:69] + "..."
                    opts.append((short, k))
            lab_list = [o[0] for o in opts]
            sel_i = st.selectbox("训练样本范围", range(len(lab_list)), format_func=lambda i: lab_list[i])
            train_fok: str | None
            mode = opts[sel_i][1]
            if mode == "all":
                train_fok = None
            elif mode == "null":
                train_fok = "__UNLABELED__"
            else:
                train_fok = mode
            if st.button("执行训练"):
                if train_fok == "__UNLABELED__":
                    ok, msg = fit_and_store_model(
                        conn, op_sel, dev_sel, unlabeled_only=True
                    )
                else:
                    ok, msg = fit_and_store_model(
                        conn, op_sel, dev_sel, feature_order_key=train_fok
                    )
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

    with tab3:
        st.markdown(
            "按 **算子 + 设备 + feature_order_key** 维度导出展平 CSV（含 `record_id`、`feature_order_key` 列）。"
        )
        pairs = list_op_device_pairs(conn)
        if not pairs:
            st.warning("暂无数据可导出。")
        else:
            labels = [f"{a} @ {b}" for a, b in pairs]
            idx = st.selectbox("导出对象", range(len(labels)), format_func=lambda i: labels[i], key="ex")
            op_sel, dev_sel = pairs[idx]
            ek = list_record_export_keys(conn, op_sel, dev_sel)
            ex_opts: list[tuple[str, str, str | None, bool]] = []
            ex_opts.append(("全部记录（不按 key 筛选）", "all", None, False))
            if None in ek:
                ex_opts.append(("仅未标注 (feature_order_key 为空)", "null", None, True))
            for k in ek:
                if k is not None:
                    ex_opts.append(
                        (
                            k if len(k) <= 96 else k[:93] + "...",
                            "key",
                            k,
                            False,
                        )
                    )
            ex_labels = [o[0] for o in ex_opts]
            ex_i = st.selectbox(
                "feature_order_key 范围",
                range(len(ex_labels)),
                format_func=lambda i: ex_labels[i],
                key="ex_fok",
            )
            mode = ex_opts[ex_i][1]
            if mode == "all":
                header, rows = export_records_flat_csv_rows(conn, op_sel, dev_sel)
                fname = f"ocm_{export_filename_suffix(op_sel, dev_sel, 'all')}.csv"
            elif mode == "null":
                header, rows = export_records_flat_csv_rows(
                    conn, op_sel, dev_sel, unlabeled_only=True
                )
                fname = f"ocm_{export_filename_suffix(op_sel, dev_sel, 'unlabeled')}.csv"
            else:
                k = ex_opts[ex_i][2]
                assert k is not None
                header, rows = export_records_flat_csv_rows(conn, op_sel, dev_sel, k)
                fname = f"ocm_{export_filename_suffix(op_sel, dev_sel, k)}.csv"
            if header:
                df = pd.DataFrame(rows, columns=header)
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                st.download_button(
                    "下载 CSV",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name=fname,
                    mime="text/csv",
                    key="dl_csv",
                )
                st.dataframe(df, use_container_width=True)

    with tab4:
        st.markdown(
            "将离线训练得到的 **model_payload**（XGBoost `save_raw('json')` 文本）"
            "与 **feature_order**（JSON 数组，特征名顺序与训练时一致）粘贴覆盖数据库。"
        )
        op_m = st.text_input("op_name", value="nn::conv2d_nchw_fp32", key="m_op")
        dev_m = st.text_input("device", value="NVIDIA_RTX_4090", key="m_dev")
        payload = st.text_area("model_payload（JSON 字符串）", height=200)
        feat_order = st.text_area(
            'feature_order（JSON 数组，如 ["C","H","W","is_contiguous","memory_stride_0"]）',
            height=100,
        )
        if st.button("覆盖 models 表"):
            try:
                order = json.loads(feat_order)
                if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
                    raise ValueError("feature_order 必须是非空字符串数组")
                fk = upsert_model(conn, op_m, dev_m, payload.strip(), order)
                st.success(f"已更新模型。主键 feature_order_key=`{fk}`")
            except Exception as e:
                st.error(str(e))

    with tab5:
        st.markdown("编译器侧推理：查询 `models`，有则 `predict`，无则返回 None。")
        op_p = st.text_input("op_name", value="nn::conv2d_nchw_fp32", key="p_op")
        dev_p = st.text_input("device", value="NVIDIA_RTX_4090", key="p_dev")

        with st.expander("params 模板（可选）", expanded=False):
            tpls_i = list_param_templates(conn)
            names_i = [t["name"] for t in tpls_i]
            i1, i2, i3 = st.columns([2, 1, 1])
            with i1:
                pick_i = st.selectbox("选择模板", ["—"] + names_i, key="tpl_pick_infer")
            with i2:
                st.write("")
                if st.button("加载到编辑区", key="btn_tpl_load_infer"):
                    if pick_i != "—":
                        t = get_param_template_by_name(conn, pick_i)
                        if t:
                            st.session_state.infer_params = json.dumps(
                                t["params"], ensure_ascii=False, separators=(",", ":")
                            )
                            st.rerun()
            with i3:
                st.write("")
            in1, in2 = st.columns([3, 1])
            with in1:
                st.text_input(
                    "将当前 params 保存为模板",
                    key="tpl_save_name_infer",
                    placeholder="模板名称",
                )
            with in2:
                st.write("")
                st.write("")
                if st.button("保存为模板", key="btn_tpl_save_infer"):
                    try:
                        p = json.loads(st.session_state.infer_params)
                        if not isinstance(p, dict):
                            raise ValueError("params 必须是 JSON 对象")
                        nm = (st.session_state.get("tpl_save_name_infer") or "").strip()
                        if not nm:
                            raise ValueError("请填写模板名称")
                        save_param_template(conn, nm, p)
                        st.success(f"已保存模板「{nm}」")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.text_area(
            "params (JSON)",
            height=140,
            key="infer_params",
        )
        mods = list_models_for_op_device(conn, op_p, dev_p)
        pred_fok: str | None = None
        if len(mods) > 1:
            pred_fok = st.selectbox(
                "模型变体 (feature_order_key)",
                [m["feature_order_key"] for m in mods],
                key="infer_model_variant",
            )
        if st.button("预测"):
            try:
                pdict = json.loads(st.session_state.infer_params)
                pred = predict_latency(
                    conn,
                    op_p,
                    dev_p,
                    pdict,
                    feature_order_key=pred_fok,
                )
                if pred is None:
                    st.info("无模型：返回 None（可回退到真实 Benchmark）。")
                else:
                    st.metric("预测 latency (ms)", f"{pred:.6f}")
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.markdown("**不经过数据库**：手动粘贴 model_payload + feature_order 试算。")
        pay2 = st.text_area("model_payload", height=120, key="p2_pay")
        ord2 = st.text_area("feature_order JSON", height=80, key="p2_ord")
        if st.button("直接预测"):
            try:
                order = json.loads(ord2)
                pdict = json.loads(st.session_state.infer_params)
                if not isinstance(order, list):
                    raise ValueError("feature_order 无效")
                val = predict_with_booster_json(pay2.strip(), order, pdict)
                st.metric("预测 latency (ms)", f"{val:.6f}")
            except Exception as e:
                st.error(str(e))

    conn.close()


if __name__ == "__main__":
    main()
