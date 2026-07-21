import os
from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import polars as pl
import numpy as np

# --- Definimos propiedades disponibles ---
propiedades = ['Clorofila', 'OxigenoDisuelto', 'Salinidad', 'Temperatura']

# --- Unidades por propiedad ---
unidades = {
    'Clorofila':      'mg/m³',
    'OxigenoDisuelto': 'mg/L',
    'Salinidad':      'PSU',
    'Temperatura':    '°C',
}

def getPlot(boya, group_by_depth=False, group_all=False):
    # Si no hay datasets todavía (PVC vacío al arrancar), devuelve figura vacía
    if not any(os.path.exists(f'datasets/{p}_series.csv') for p in propiedades):
        fig = go.Figure()
        fig.update_layout(title="Cargando datos...", height=500)
        return fig

    fig = go.Figure()
    list_buttons = []

    # Guardamos a qué propiedad pertenece cada traza (en el mismo orden en que se añaden)
    trace_props = []

    # Grupos si agrupamos por profundidad
    if group_by_depth:
        # 3 colores para los tres grupos
        custom_colors_local = ["#F2BF6B", "#6BE76C", "#6B6FF0"]  # 0–2, 2–4, >4
        depth_groups = {
            "0-2":  ['-0.5', '-1.0', '-1.5'],
            "2-4": ['-2.0', '-2.5', '-3.0', '-3.5'],
            "4-6":  ['-4.0', '-4.5', '-5.0']
        }
    elif group_all:
        custom_colors_local = ["#6B6FF0"]  # un único color para la media global
    else:
        # 10 colores para profundidades individuales
        custom_colors_local = [
            "#F27F6B","#F2BF6B","#F2F26B","#BEE76B","#6BE76C",
            "#6BE7B8","#6BE7E7","#64B1ED","#6B6FF0","#BA8FF2"
        ]

    # 1) Añadir todas las trazas (inicialmente invisibles)
    for p in propiedades:
        df = pl.read_csv(f'datasets/{p}_series.csv', try_parse_dates=True)
        if df['Date'].dtype in (pl.Utf8, pl.String):
            df = df.with_columns(pl.col('Date').str.to_datetime(strict=False))
        df = df.sort('Date')
        depth_cols = [c for c in df.columns if c != 'Date']
        df = df.with_columns([
            pl.col(c).cast(pl.Float64).interpolate() for c in depth_cols
        ])

        # Calculo de los valores según el modo de agrupación
        if group_by_depth:
            exprs = [pl.col('Date')]
            for name, cols in depth_groups.items():
                present = [c for c in cols if c in df.columns]
                if present:
                    exprs.append(pl.mean_horizontal([pl.col(c) for c in present]).alias(name))
                else:
                    exprs.append(pl.lit(None, dtype=pl.Float64).alias(name))
            df = df.select(exprs)
        elif group_all:
            all_cols = [c for c in df.columns if c != 'Date']
            df = df.select([
                pl.col('Date'),
                pl.mean_horizontal([pl.col(c) for c in all_cols]).alias('Todas las profundidades')
            ])

        plot_cols = [c for c in df.columns if c != 'Date']
        for i, c in enumerate(plot_cols):
            color = custom_colors_local[i % len(custom_colors_local)]
            fig.add_trace(go.Scatter(
                x=df['Date'].to_list(), y=df[c].to_list(),
                name=(c + 'm') if not (group_by_depth or group_all) else c,
                legendgroup=p, legendgrouptitle=dict(text=f"{p} ({unidades[p]})"),
                visible=False,
                line=dict(color=color, width=2)
            ))
            trace_props.append(p)


    # 2) Botones: visibilidad robusta (True sólo para las trazas de esa propiedad)
    for p in propiedades:
        visibles = [prop == p for prop in trace_props]
        list_buttons.append(dict(
            label=f"{p} ({unidades[p]})",
            method="update",
            args=[{"visible": visibles},
                  {"title": f"<b>{boya} — {p} ({unidades[p]})</b>",
                   "yaxis.title": unidades[p]}]
        ))

    # 3) Layout
    fig.update_layout(
        updatemenus=[dict(buttons=list_buttons, font=dict(size=16))],
        legend=dict(groupclick="toggleitem"),
        title=f"<b>{boya} — {p} ({unidades[p]})</b>",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title=unidades[propiedades[0]]),
        uirevision="keep",
        height=500,
        # etiqueta pequeña que muestre el estado del switch
        annotations=[dict(
            xref="paper", yref="paper", x=0, y=1.08, xanchor="left",
            text=f"Agrupación por profundidad: {'Sí' if group_by_depth else 'No'}",
            showarrow=False, font=dict(size=12)
        )]
    )

    # 4) Estado inicial: primera propiedad encendida + título inicial
    first_prop = propiedades[0]
    initial_visible = [prop == first_prop for prop in trace_props]
    for tr, v in zip(fig.data, initial_visible):
        tr.visible = v
    fig.update_layout(title=f"<b>{boya} — {first_prop} ({unidades[first_prop]})</b>")

    # 5) Línea de media inicial
    ys = []
    for tr in fig.data:
        if tr.visible in [True, None]:
            ys.extend([float(v) for v in (tr.y or []) if v is not None and v == v])

    if ys:
        ymean = float(np.nanmean(ys))
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1,
            yref="y", y0=ymean, y1=ymean,
            line=dict(dash="longdash")
        )
        fig.add_annotation(
            xref="paper", x=1, xanchor="right", y=ymean,
            text=f"Media del período: {ymean:.2f}",
            showarrow=False, bgcolor="white"
        )

    return fig


# Aplicación Dash
_prefix = os.environ.get("DASH_PATHNAME_PREFIX", "/")
# En Dash 4, las rutas Flask siempre se registran en "/", independientemente
# de requests_pathname_prefix. routes_pathname_prefix controla las URLs del
# cliente (renderer). nginx debe quitar el prefijo antes de pasar al pod.
app = Dash(__name__, routes_pathname_prefix=_prefix, requests_pathname_prefix=_prefix)
app.title = "Visor Columna de Agua"

app.layout = html.Div([
    # Título del layout
    html.H3("Datos de la columna de agua del Mar Menor"),

    # Casilla: agrupar por profundidad
    dcc.Checklist(
        id='chk-group-depth',
        options=[{'label': ' Agrupar por profundidades (0-2 / 2-4 / 4-6 m)', 'value': 'on'}],
        value=[],
        style={'margin': '8px 0'}
    ),

    # Casilla: agrupar todas las profundidades
    dcc.Checklist(
        id='chk-group-all',
        options=[{'label': ' Agrupar todas las profundidades', 'value': 'on'}],
        value=[],
        style={'margin': '8px 0'}
    ),

    # Casilla: mostrar segunda gráfica
    dcc.Checklist(
        id='toggle-second-graph',
        options=[{'label': ' Mostrar segunda gráfica para comparar otra propiedad', 'value': 'on'}],
        value=[],  # desactivada por defecto
        style={'margin': '8px 0'}
    ),

    # Gráfica 1
    dcc.Graph(
        id='grafico-columnaagua-1',
        figure=getPlot('Datos Agregados', group_by_depth=False),
        style={'height': '500px', 'margin-bottom': '24px'}
    ),

    # Contenedor 2ª gráfica (se oculta si no está marcada)
    html.Div(
        id='wrap-second',
        children=[
            dcc.Graph(
                id='grafico-columnaagua-2',
                figure=getPlot('Datos Agregados', group_by_depth=False),
                style={'height': '500px'}
            )
        ],
        style={'display': 'none'}
    ),

    # Pie de página
    html.Footer(
        html.P([
            "The data gathered comes from Polytechnic University of Cartagena (UPCT) at ",
            html.A("https://marmenor.upct.es/embed/l2", href="https://marmenor.upct.es/embed/l2", target="_blank"),
            " and Murcian Institute of Agricultural and Environmental Research and Development (IMIDA) at ",
            html.A("https://idearm.imida.es/cgi/siomctdmarmenor/", href="https://idearm.imida.es/cgi/siomctdmarmenor/", target="_blank"),
            ". The latter is no longer available due to the end of the project related to that website."
        ]),
        style={
            'marginTop': '32px',
            'padding': '16px',
            'borderTop': '1px solid #ccc',
            'fontSize': '13px',
            'color': '#666',
            'textAlign': 'justify'
        }
    )
])

# ---------- ÚNICO callback maestro (sin salidas duplicadas) ----------
@app.callback(
    Output('wrap-second', 'style'),
    Output('grafico-columnaagua-1', 'figure'),
    Output('grafico-columnaagua-2', 'figure'),
    Output('chk-group-depth', 'value'),
    Output('chk-group-all', 'value'),
    Input('chk-group-depth', 'value'),
    Input('chk-group-all', 'value'),
    Input('toggle-second-graph', 'value'),
    Input('grafico-columnaagua-1', 'relayoutData'),
    Input('grafico-columnaagua-2', 'relayoutData'),
    Input('grafico-columnaagua-1', 'restyleData'),
    Input('grafico-columnaagua-2', 'restyleData'),
    State('grafico-columnaagua-1', 'figure'),
    State('grafico-columnaagua-2', 'figure'),
    prevent_initial_call=True
)
def master_update(chk_group, chk_group_all, toggle_second, relayout1, relayout2, restyle1, restyle2, fig1, fig2):

    # --- 0) Estado de las casillas ---
    grouped     = 'on' in (chk_group     or [])
    grouped_all = 'on' in (chk_group_all or [])
    show_second = 'on' in (toggle_second or [])
    wrap_style  = {'display': 'block'} if show_second else {'display': 'none'}

    # Exclusión mutua: la casilla recién activada gana, la otra se apaga
    out_chk_depth = chk_group     or []
    out_chk_all   = chk_group_all or []
    trigger = ctx.triggered_id
    if trigger == 'chk-group-depth' and grouped:
        out_chk_all = []
        grouped_all = False
    elif trigger == 'chk-group-all' and grouped_all:
        out_chk_depth = []
        grouped = False

    # --- 1) Determinar qué provocó el callback ---

    # --- 2) Si cambió el modo de agrupación → reconstruimos ambas figuras
    if trigger in ('chk-group-depth', 'chk-group-all'):
        # rangos previos si existen
        xr1 = fig1.get('layout', {}).get('xaxis', {}).get('range')
        xr2 = fig2.get('layout', {}).get('xaxis', {}).get('range')

        fig1 = getPlot('Datos Agregados', group_by_depth=grouped, group_all=grouped_all).to_dict()
        fig2 = getPlot('Datos Agregados', group_by_depth=grouped, group_all=grouped_all).to_dict()

        # preservar rangos previos
        if xr1:
            fig1.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr1
        if xr2:
            fig2.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr2

    # --- 3) Si se activó la 2ª gráfica ahora mismo
    if trigger == 'toggle-second-graph' and show_second:
        fig2 = getPlot('Datos Agregados', group_by_depth=grouped, group_all=grouped_all).to_dict()
        xr1 = fig1.get('layout', {}).get('xaxis', {}).get('range')
        if xr1:
            fig2.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr1

    # --- 4) Sincronización de rangos X entre gráficas (según quién provocó el relayout)
    def extract_range(relayout):
        if not relayout:
            return None
        if 'xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout:
            return [relayout['xaxis.range[0]'], relayout['xaxis.range[1]']]
        if 'xaxis.range' in relayout and isinstance(relayout['xaxis.range'], list):
            return list(relayout['xaxis.range'])
        if 'xaxis.autorange' in relayout:
            return 'autorange'
        return None

    def apply_range(fig, rng):
        fig.setdefault('layout', {}).setdefault('xaxis', {})
        if rng == 'autorange':
            fig['layout']['xaxis']['autorange'] = True
            fig['layout']['xaxis'].pop('range', None)
        elif isinstance(rng, list) and len(rng) == 2:
            fig['layout']['xaxis']['autorange'] = False
            fig['layout']['xaxis']['range'] = rng
        return fig

    rng1 = extract_range(relayout1)
    rng2 = extract_range(relayout2)
    # Si el usuario movió 1 entonces sincroniza 2
    if ctx.triggered_id == 'grafico-columnaagua-1' and rng1 is not None and show_second:
        fig2 = apply_range(fig2, rng1)
    # Si el usuario movió 2 entonces sincroniza 1
    if ctx.triggered_id == 'grafico-columnaagua-2' and rng2 is not None:
        fig1 = apply_range(fig1, rng2)
    # 5) Mantener siempre uirevision (evita "rebotes" del slider/zoom) ---
    for f in (fig1, fig2):
        f.setdefault('layout', {}).update({'uirevision': 'keep'})

    # --- 6) Recalcular MEDIA visible + rango Y para cada figura ---
    def recompute_mean_and_y(fig, relayout):

        def parse_dt64(s):
            """Convierte cualquier cadena de fecha a numpy datetime64[ns]."""
            s = str(s).strip().replace(' ', 'T')
            for suffix in ('+00:00', '+0000', 'Z'):
                s = s.replace(suffix, '')
            return np.datetime64(s, 'ns')

        # Determinar rango X visible
        xs_raw = xe_raw = None
        if relayout:
            if 'xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout:
                xs_raw = relayout['xaxis.range[0]']
                xe_raw = relayout['xaxis.range[1]']
            elif 'xaxis.range' in relayout and isinstance(relayout['xaxis.range'], list):
                xs_raw, xe_raw = relayout['xaxis.range'][0], relayout['xaxis.range'][1]
        if xs_raw is None:
            xr = fig.get('layout', {}).get('xaxis', {}).get('range')
            if xr:
                xs_raw, xe_raw = xr[0], xr[1]
            else:
                all_x = []
                for tr in fig.get('data', []):
                    raw = tr.get('x') or []
                    flat = [v for sub in raw for v in sub] if raw and isinstance(raw[0], (list, tuple)) else list(raw)
                    all_x.extend(flat)
                if all_x:
                    xs_raw, xe_raw = min(all_x), max(all_x)

        # Recoger valores Y visibles dentro del rango — Python puro, sin pandas
        ys = []
        if xs_raw is not None and xe_raw is not None:
            xs_np = parse_dt64(xs_raw)
            xe_np = parse_dt64(xe_raw)
            for tr in fig.get('data', []):
                vis = tr.get('visible', True)
                if vis is False or vis == 'legendonly':
                    continue
                x_raw = tr.get('x') or []
                y_raw = tr.get('y') or []
                # Aplanar listas anidadas si las hay
                if x_raw and isinstance(x_raw[0], (list, tuple)):
                    x_raw = [v for sub in x_raw for v in sub]
                if y_raw and isinstance(y_raw[0], (list, tuple)):
                    y_raw = [v for sub in y_raw for v in sub]
                for xv, yv in zip(x_raw, y_raw):
                    try:
                        if xs_np <= parse_dt64(xv) <= xe_np and yv is not None:
                            ys.append(float(yv))
                    except Exception:
                        continue

        # Limpiar shapes/annotations de media y redibujar
        fig.setdefault('layout', {})
        fig['layout']['shapes'] = []
        fig.setdefault('layout', {}).setdefault('annotations', [])
        fig['layout']['annotations'] = [
            a for a in fig['layout']['annotations']
            if not a.get('text', '').startswith('Media del período:')
        ]

        if ys:
            ymean = float(np.nanmean(ys))
            fig['layout']['shapes'].append({
                'type': 'line', 'xref': 'paper', 'x0': 0, 'x1': 1,
                'yref': 'y', 'y0': ymean, 'y1': ymean,
                'line': {'dash': 'dash', 'width': 2}
            })
            fig['layout']['annotations'].append({
                'xref': 'paper', 'x': 1, 'xanchor': 'right', 'y': ymean,
                'text': f'Media del período: {ymean:.2f}',
                'showarrow': False, 'bgcolor': 'white'
            })
            ymin = float(np.nanmin(ys))
            ymax = float(np.nanmax(ys))
            if np.isfinite(ymin) and np.isfinite(ymax):
                if ymin == ymax:
                    delta = 1.0 if ymax == 0 else abs(ymax) * 0.05
                    fig.setdefault('layout', {}).setdefault('yaxis', {})
                    fig['layout']['yaxis']['autorange'] = False
                    fig['layout']['yaxis']['range'] = [ymin - delta, ymax + delta]
                else:
                    pad = 0.05
                    span = ymax - ymin
                    fig.setdefault('layout', {}).setdefault('yaxis', {})
                    fig['layout']['yaxis']['autorange'] = False
                    fig['layout']['yaxis']['range'] = [ymin - span * pad, ymax + span * pad]
        else:
            fig.setdefault('layout', {}).setdefault('yaxis', {})
            fig['layout']['yaxis']['autorange'] = True
            fig['layout']['yaxis'].pop('range', None)

        # Parche Plotly 6 (_template)
        rs = fig.get('layout', {}).get('xaxis', {}).get('rangeslider', {})
        if isinstance(rs, dict) and 'yaxis' in rs and isinstance(rs['yaxis'], dict):
            rs['yaxis'].pop('_template', None)

        return fig

    fig1 = recompute_mean_and_y(fig1, relayout1)
    if show_second:
        fig2 = recompute_mean_and_y(fig2, relayout2)

    return wrap_style, fig1, fig2, out_chk_depth, out_chk_all


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)
