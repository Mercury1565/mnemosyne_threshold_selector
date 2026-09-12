## Step 1: count harmful frames in a region
For a given (t_l, t_h), define the two regions:

$$C(t_h) = \{i : \text{score}_i \geq t_h\} \qquad \text{(copy region)}$$
$$I(t_l, t_h) = \{i \in D_{\text{IA}} : t_l \leq \text{score}_i < t_h\} \qquad \text{(IA region)}$$

where $D_{\text{IA}}$ is the subset of ~781 frames that actually have a measured IA-reuse outcome (the other ~3,114 frames don't have one, so they can never contribute to ia_risk).

For each region, count:

$$n = |\text{region}| \qquad k = |\{i \in \text{region} : \text{loss}_i > \text{TAU}\}|$$

For the copy region: $\text{loss}_i = (\text{loss}_{\text{copy}})_i$ (the inferred proxy)
For the IA region: $\text{loss}_i = (\text{loss}_{\text{ia}})_i$ (the measured gap)

## Step 2: the Clopper-Pearson upper bound
Given that $(n, k)$ pair, the risk number is:

$$\text{risk} = F^{-1}_{\text{Beta}(k+1,\ n-k)}(\text{CONF})$$

i.e. the $\text{CONF}$-quantile (90%, by default) of a $\text{Beta}(k{+}1, n{-}k)$ distribution — the value $p$ such that "if the true harm rate were $p$, seeing only $k$ or fewer harmful frames out of $n$ would be no more than a $(1-\text{CONF})$ coincidence." Edge case: if $n=0$, $\text{risk}=1$ (no evidence, so assume the worst).

Put together

$$\text{copy\_risk}(t_h) = F^{-1}_{\text{Beta}(k_c+1,\ n_c-k_c)}(\text{CONF}), \quad n_c = |C(t_h)|,\ k_c = |\{i \in C(t_h): (\text{loss}_{\text{copy}})_i > \text{TAU}\}|$$

$$\text{ia\_risk}(t_l, t_h) = F^{-1}_{\text{Beta}(k_a+1,\ n_a-k_a)}(\text{CONF}), \quad n_a = |I(t_l,t_h)|,\ k_a = |\{i \in I(t_l,t_h): (\text{loss}_{\text{ia}})_i > \text{TAU}\}|$$

Same functional form both times — only which frames fall into $(n, k)$, and which loss column defines "harmful," differ between the two.
