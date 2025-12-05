from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Definimos propiedades disponibles ---
propiedades = ['Clorofila', 'OxigenoDisuelto', 'Salinidad', 'Temperatura']

def getPlot(boya, group_by_depth=False):
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
    else:
        # 10 colores para profundidades individuales
        custom_colors_local = [
            "#F27F6B","#F2BF6B","#F2F26B","#BEE76B","#6BE76C",
            "#6BE7B8","#6BE7E7","#64B1ED","#6B6FF0","#BA8FF2"
        ]

    # 1) Añadir todas las trazas (inicialmente invisibles)
    for p in propiedades:
        # Leemos los csv obtenidos en LecturaDatosTratados.ipynb
        df = pd.read_csv(f'boyas/{p}_series.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Interpolación lineal para rellenar NaN
        df[df.columns[1:]] = df[df.columns[1:]].interpolate(method='linear')

        # Calculo de los valores si se agrupa por profundidad
        if group_by_depth:
            df_grouped = df[['Date']].copy()
            for name, cols in depth_groups.items():
                cols_present = [c for c in cols if c in df.columns]
                df_grouped[name] = df[cols_present].mean(axis=1) if cols_present else pd.NA
            df = df_grouped

        # Añadimos líneas para cada profundidad (una por columna)
        for i, c in enumerate(df.columns[1:]):
            color = custom_colors_local[i % len(custom_colors_local)]
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[c],
                name=(c + 'm') if not group_by_depth else c,
                legendgroup=p, legendgrouptitle=dict(text=p),
                visible=False,
                line=dict(color=color, width=2)
            ))
            trace_props.append(p)


    # 2) Botones: visibilidad robusta (True sólo para las trazas de esa propiedad)
    for p in propiedades:
        visibles = [prop == p for prop in trace_props]
        list_buttons.append(dict(
            label=p,
            method="update",
            args=[{"visible": visibles},
                  {"title": f"<b>{boya} — {p}</b>"}]
        ))

    # 3) Layout
    fig.update_layout(
        updatemenus=[dict(buttons=list_buttons, font=dict(size=16))],
        legend=dict(groupclick="toggleitem"),
        title=f"<b>{boya} — {p}</b>",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
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
    fig.update_layout(title=f"<b>{boya} — {first_prop}</b>")

    # 5) Línea de media inicial
    ys = []
    for tr in fig.data:
        if tr.visible in [True, None]:
            ys.extend(pd.to_numeric(pd.Series(tr.y), errors='coerce').tolist())

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
app = Dash(__name__, requests_pathname_prefix="/visorboyas/")
app.title = "Visor Boyas"

app.layout = html.Div([
    # Título del layout
    html.H3("Datos de la boyas del Mar Menor"),

    # Casilla: agrupar por profundidad
    dcc.Checklist(
        id='chk-group-depth',
        options=[{'label': ' Agrupar por profundidades (0-2 / 2-4 / 4-6)', 'value': 'on'}],
        value=[],  # activado por defecto 
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
        id='grafico-boyas-1',
        figure=getPlot('CTD Agregados', group_by_depth=False),
        style={'height': '500px', 'margin-bottom': '24px'}
    ),

    # Contenedor 2ª gráfica (se oculta si no está marcada)
    html.Div(
        id='wrap-second',
        children=[
            dcc.Graph(
                id='grafico-boyas-2',
                figure=getPlot('CTD Agregados', group_by_depth=False),
                style={'height': '500px'}
            )
        ],
        style={'display': 'none'}
    )
])

# ---------- ÚNICO callback maestro (sin salidas duplicadas) ----------
@app.callback(
    Output('wrap-second', 'style'),
    Output('grafico-boyas-1', 'figure'),
    Output('grafico-boyas-2', 'figure'),
    Input('chk-group-depth', 'value'),
    Input('toggle-second-graph', 'value'),
    Input('grafico-boyas-1', 'relayoutData'),
    Input('grafico-boyas-2', 'relayoutData'),
    Input('grafico-boyas-1', 'restyleData'),
    Input('grafico-boyas-2', 'restyleData'),
    State('grafico-boyas-1', 'figure'),
    State('grafico-boyas-2', 'figure'),
    prevent_initial_call=True
)
def master_update(chk_group, toggle_second, relayout1, relayout2, restyle1, restyle2, fig1, fig2):

    # --- 0) Estado de las casillas ---
    grouped = 'on' in (chk_group or [])
    show_second = 'on' in (toggle_second or [])
    wrap_style = {'display': 'block'} if show_second else {'display': 'none'}

    # --- 1) Determinar qué provocó el callback ---
    trigger = ctx.triggered_id

    # --- 2) Si cambió el modo de agrupación → reconstruimos ambas figuras (preservando rangos X si los hay)
    if trigger == 'chk-group-depth':
        # rangos previos si existen
        xr1 = fig1.get('layout', {}).get('xaxis', {}).get('range')
        xr2 = fig2.get('layout', {}).get('xaxis', {}).get('range')

        fig1 = getPlot('CTD Agregados', group_by_depth=grouped).to_dict()
        fig2 = getPlot('CTD Agregados', group_by_depth=grouped).to_dict()

        # preservar rangos previos
        if xr1:
            fig1.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr1
        if xr2:
            fig2.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr2

    # --- 3) Si se activó la 2ª gráfica ahora mismo - inicialízala y (opcional) copia rango de la primera
    if trigger == 'toggle-second-graph' and show_second:
        fig2 = getPlot('CTD Agregados', group_by_depth=grouped).to_dict()
        xr1 = fig1.get('layout', {}).get('xaxis', {}).get('range')
        if xr1:
            fig2.setdefault('layout', {}).setdefault('xaxis', {})['range'] = xr1

    # --- 4) Sincronización de rangos X entre gráficas (según quién provocó el relayout)
    def extract_range(relayout):
        if not relayout:
            return None
        if 'xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout:
            return [pd.to_datetime(relayout['xaxis.range[0]']),
                    pd.to_datetime(relayout['xaxis.range[1]'])]
        if 'xaxis.range' in relayout and isinstance(relayout['xaxis.range'], list):
            return [pd.to_datetime(relayout['xaxis.range'][0]),
                    pd.to_datetime(relayout['xaxis.range'][1])]
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
    if ctx.triggered_id == 'grafico-boyas-1' and rng1 is not None and show_second:
        fig2 = apply_range(fig2, rng1)
    # Si el usuario movió 2 entonces sincroniza 1
    if ctx.triggered_id == 'grafico-boyas-2' and rng2 is not None:
        fig1 = apply_range(fig1, rng2)

    # --- 5) Mantener siempre uirevision (evita “rebotes” del slider/zoom) ---
    for f in (fig1, fig2):
        f.setdefault('layout', {}).update({'uirevision': 'keep'})

    # --- 6) Recalcular MEDIA visible + rango Y para cada figura ---
    def recompute_mean_and_y(fig, relayout):
        # rango visible
        xs = xe = None
        if relayout:
            if 'xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout:
                xs = pd.to_datetime(relayout['xaxis.range[0]'])
                xe = pd.to_datetime(relayout['xaxis.range[1]'])
            elif 'xaxis.range' in relayout and isinstance(relayout['xaxis.range'], list):
                xs = pd.to_datetime(relayout['xaxis.range'][0])
                xe = pd.to_datetime(relayout['xaxis.range'][1])
        if xs is None or xe is None:
            xr = fig.get('layout', {}).get('xaxis', {}).get('range')
            if xr:
                xs, xe = pd.to_datetime(xr[0]), pd.to_datetime(xr[1])
            else:
                xmins, xmaxs = [], []
                for tr in fig.get('data', []):
                    if tr.get('x'):
                        xmins.append(pd.to_datetime(min(tr['x'])))
                        xmaxs.append(pd.to_datetime(max(tr['x'])))
                if xmins and xmaxs:
                    xs, xe = min(xmins), max(xmaxs)

        # valores visibles
        ys = []
        if xs is not None and xe is not None:
            for tr in fig.get('data', []):
                vis = tr.get('visible', True)
                if vis is False or vis == 'legendonly':
                    continue
                x = pd.to_datetime(pd.Series(tr.get('x', [])))
                y = pd.to_numeric(pd.Series(tr.get('y', [])), errors='coerce')
                mask = (x >= xs) & (x <= xe)
                ys.extend(y[mask].tolist())

        # limpiar shapes/annotations de media y volver a dibujar
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

            # rango Y manual con padding
            ymin = float(np.nanmin(ys)); ymax = float(np.nanmax(ys))
            if np.isfinite(ymin) and np.isfinite(ymax):
                if ymin == ymax:
                    delta = 1.0 if ymax == 0 else abs(ymax)*0.05
                    fig.setdefault('layout', {}).setdefault('yaxis', {})
                    fig['layout']['yaxis']['autorange'] = False
                    fig['layout']['yaxis']['range'] = [ymin - delta, ymax + delta]
                else:
                    pad = 0.05; span = ymax - ymin
                    fig.setdefault('layout', {}).setdefault('yaxis', {})
                    fig['layout']['yaxis']['autorange'] = False
                    fig['layout']['yaxis']['range'] = [ymin - span*pad, ymax + span*pad]
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

    return wrap_style, fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8050)

