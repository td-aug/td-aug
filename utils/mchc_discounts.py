"""Discount conversions shared by MCHC datasets and agents."""


def resolve_mchc_discounts(config):
    """Return local/final discounts in the MCHC sampling time unit."""
    local_discount = float(config["discount"])
    final_discount = float(config["final_discount"])
    sampling_unit = config.get("mchc_sampling_unit", "primitive")

    if sampling_unit not in ("primitive", "chunk"):
        raise ValueError(
            "mchc_sampling_unit must be 'primitive' or 'chunk', got "
            f"{sampling_unit!r}."
        )
    if not 0.0 <= local_discount < 1.0:
        raise ValueError("MCHC discount must be in [0, 1).")
    if not 0.0 <= final_discount < 1.0:
        raise ValueError("MCHC final_discount must be in [0, 1).")

    if sampling_unit == "primitive":
        effective_local = local_discount
        effective_final = final_discount
    else:
        action_chunk_size = int(config["action_chunk_size"])
        bootstrap_stride = int(config.get("bootstrap_stride", 1))
        if action_chunk_size < 1 or bootstrap_stride < 1:
            raise ValueError(
                "MCHC chunk sampling requires positive action_chunk_size and "
                "bootstrap_stride."
            )
        if bootstrap_stride != action_chunk_size:
            raise ValueError(
                "mchc_sampling_unit='chunk' requires bootstrap_stride == "
                "action_chunk_size so one model transition is one action chunk; "
                f"got bootstrap_stride={bootstrap_stride} and "
                f"action_chunk_size={action_chunk_size}."
            )

        minimum_raw_discount = 1.0 - 1.0 / bootstrap_stride
        if local_discount < minimum_raw_discount - 1e-12:
            local_horizon = 1.0 / (1.0 - local_discount)
            raise ValueError(
                "Horizon conversion requires the raw local expected horizon "
                "to be at least one chunk: "
                "1 / (1 - discount) >= bootstrap_stride. Got "
                f"discount={local_discount}, expected_horizon={local_horizon}, "
                f"bootstrap_stride={bootstrap_stride}, and minimum_discount="
                f"{minimum_raw_discount}."
            )
        effective_local = max(
            0.0, 1.0 - bootstrap_stride * (1.0 - local_discount)
        )
        effective_final = max(
            0.0, 1.0 - bootstrap_stride * (1.0 - final_discount)
        )

    if not 0.0 <= effective_local < 1.0:
        raise ValueError(
            f"Effective MCHC local discount must be in [0, 1), got {effective_local}."
        )
    if not effective_local <= effective_final < 1.0:
        raise ValueError(
            "Effective MCHC discounts require local <= final < 1; got "
            f"local={effective_local}, final={effective_final}."
        )
    return effective_local, effective_final
