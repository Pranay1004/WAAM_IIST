"""
Deposition schedule.

Reproduces the paper's layer-by-layer timeline (Sec. 2, Table 3):

- 40 s of arc time per pass, 1.8 mm of height added per pass;
- an **inter-pass dwell** of 5 s or 100 s in the study grid (100–180 s in the
  experiment), with the rule that the layer start point must air-cool to 100 °C
  before the next layer;
- **unidirectional (U)** and **bidirectional (B)** torch strategies, which is what the
  paper uses to study heat accumulation;
- 25 passes over 3400 s in the long case.

The scheduler is a pure state machine: `phase(state)` returns what the solver should be
doing at the current time, so it is testable without running any physics.
"""

@enum Phase ARCON DWELL FINISHED

"""
    DepositionState

Where the schedule currently is.

- `pass`        : 1-based index of the pass in progress (0 before the first)
- `t`           : simulated time [s]
- `phase`       : `:arc` or `:dwell`
- `torch`       : (x, y) arc centre for the current pass
- `direction`   : +1 or −1 (the B strategy alternates)
- `height`      : nominal wall height so far [m]
- `n_passes`    : total passes requested
- `dwell`       : inter-pass dwell [s]
- `interpass_T` : the temperature the layer start must fall to [K]
- `pass_time`   : arc-on time per pass [s]
"""
Base.@kwdef mutable struct DepositionState
    pass::Int = 0
    t::Float64 = 0.0
    phase::Symbol = :arc
    torch::Tuple{Float64,Float64} = (0.0, 0.0)
    direction::Int = 1
    height::Float64 = 0.0
    n_passes::Int = 25
    dwell::Float64 = 100.0
    interpass_T::Float64 = 373.15      # 100 °C
    pass_time::Float64 = 40.0
    strategy::Symbol = :bidirectional   # :unidirectional | :bidirectional
    pass_start_x::Float64 = 0.0
    pass_end_x::Float64 = 0.0
    t_pass::Float64 = 0.0              # time into the current pass [s]
    dwell_elapsed::Float64 = 0.0        # time into the current dwell [s]
    interpass_met::Bool = false
end

"""
    DepositionSchedule

Timeline for a multilayer build.

Fields mirror `DepositionState` plus the geometry (`x_start`, `x_end`) the torch sweeps.
`hold_for_interpass` implements the paper's rule literally: the dwell lasts until the
layer start point has cooled to `interpass_T`, so a hot start genuinely extends the
dwell rather than being truncated by a fixed timer. That is the mechanism behind the
paper's central finding — dwell time controls heat accumulation, and accumulation
controls sagging.
"""
Base.@kwdef struct DepositionSchedule
    n_passes::Int = 25
    pass_time::Float64 = 40.0
    dwell::Float64 = 100.0
    interpass_T::Float64 = 373.15
    layer_height::Float64 = 1.8e-3
    travel_speed::Float64 = 0.0028333
    x_start::Float64 = 0.002
    x_end::Float64 = 0.014
    y::Float64 = 0.0
    strategy::Symbol = :bidirectional
    hold_for_interpass::Bool = true
end

"""
    pass_length(s) -> Float64

Arc travel length per pass [m]. With the paper's 40 s pass and 2.833 mm/s this is
**113 mm** — which is exactly why the paper's thin walls are tens of millimetres long,
and is worth stating because it is easy to assume a 40 s pass is a short bead.
"""
pass_length(s::DepositionSchedule) = s.pass_time * s.travel_speed

"""
    total_time(s) -> Float64

Nominal duration of the whole build [s]: passes plus dwells. The paper's 25-pass case is
3400 s, which is consistent with 25 × 40 s of arc plus dwell.
"""
function total_time(s::DepositionSchedule)
    n = s.n_passes
    (n * s.pass_time + (n - 1) * s.dwell, n * s.pass_time, (n - 1) * s.dwell)
end

"""
    begin_pass!(st, s) -> st

Advance to the start of a pass: set the torch to the correct end of the track, flip the
direction for the bidirectional strategy, and accumulate the layer height.
"""
function begin_pass!(st::DepositionState, s::DepositionSchedule)
    st.pass += 1
    st.phase = :arc
    st.height += s.layer_height
    st.pass_start_x = pass_time_start_x(s, st.pass)
    # A pass terminates at the far end of its own sweep. For a bidirectional even pass
    # that is x_start, not x_end -- hard-coding x_end silently made every second torch
    # trip run backwards off the track.
    st.pass_end_x = st.pass_start_x == s.x_start ? s.x_end : s.x_start
    st.torch = (st.pass_start_x, s.y)
    st.direction = st.pass_end_x >= st.pass_start_x ? 1 : -1
    st
end

# A bidirectional pass sweeps from one end to the other, so it starts where the previous
# one finished; a unidirectional pass always starts at the same end.
pass_time_start_x(s::DepositionSchedule, pass::Int) =
    s.strategy === :bidirectional && iseven(pass) ? s.x_end : s.x_start

"""
    arc_position(st, s) -> Float64

Current torch x during the pass, taking the direction into account.
"""
arc_position(st::DepositionState, s::DepositionSchedule) =
    st.pass_start_x + st.direction * min(st.t_pass, s.pass_time) * s.travel_speed

"""
    step_schedule!(st, s, dt; T_at_start = nothing) -> st

Advance the schedule by one timestep `dt`. Pure bookkeeping — no physics, so it is
directly unit-testable against the paper's 3400 s timeline.
"""
function step_schedule!(st::DepositionState, s::DepositionSchedule, dt::Real;
                        T_at_start = nothing)
    st.t += dt
    if st.phase === :arc
        st.t_pass += dt
        if st.t_pass >= s.pass_time
            st.t_pass = 0.0
            st.phase = :dwell
            st.dwell_elapsed = 0.0
            st.interpass_met = !(s.hold_for_interpass && T_at_start !== nothing &&
                                 T_at_start > s.interpass_T)
        end
    elseif st.phase === :dwell
        st.dwell_elapsed += dt
        done = st.dwell_elapsed >= s.dwell
        if s.hold_for_interpass && T_at_start !== nothing
            done = done || (T_at_start <= s.interpass_T && st.dwell_elapsed >= s.dwell)
        end
        st.interpass_met = done
        if done && st.pass < s.n_passes
            begin_pass!(st, s)
        elseif done
            st.phase = :finished
        end
    end
    st
end

"""
    paper_cases() -> Vector{NamedTuple}

The paper's eight experimental/simulated cases (Table 3).

`S-` = ER70S6, `SS-` = ER316L, `U` = unidirectional, `B` = bidirectional; the number is
the dwell time in seconds.
"""
function paper_cases()
    [(:S_U5,    "ER70S6", :unidirectional,   5.0,  10),
     (:S_B5,    "ER70S6", :bidirectional,   5.0,  10),
     (:S_U100,  "ER70S6", :unidirectional, 100.0,  10),
     (:S_B100,  "ER70S6", :bidirectional, 100.0, 25),
     (:SS_U5,   "ER316L", :unidirectional,   5.0,  10),
     (:SS_B5,   "ER316L", :bidirectional,   5.0,  10),
     (:SS_U100, "ER316L", :unidirectional, 100.0,  10),
     (:SS_B100, "ER316L", :bidirectional, 100.0, 25)]
end

"""
    sagging_prediction() -> NamedTuple

The paper's measured outcomes, used as the validation target.

`pass` is the pass at which collapse occurs (`0` = stable), `t_sag` the time in seconds
(`NaN` = stable), `thickness_mm` the effective wall thickness (sim / experiment).
"""
sagging_prediction() = (
    S_U5    = (pass = 4, t_sag = 148.0,  thickness = (6.2, NaN)),
    S_B5    = (pass = 4, t_sag = 145.0,  thickness = (6.2, NaN)),
    S_U100  = (pass = 0, t_sag = NaN,    thickness = (NaN, NaN)),
    S_B100  = (pass = 0, t_sag = NaN,    thickness = (6.2, 5.3)),
    SS_U5   = (pass = 4, t_sag = 142.0,  thickness = (6.2, NaN)),
    SS_B5   = (pass = 3, t_sag = 119.0,  thickness = (6.2, NaN)),
    SS_U100 = (pass = 0, t_sag = NaN,    thickness = (NaN, NaN)),
    SS_B100 = (pass = 0, t_sag = NaN,    thickness = (6.3, 6.2)),
)