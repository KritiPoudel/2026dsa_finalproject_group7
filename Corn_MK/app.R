library(shiny)
library(ggplot2)
library(dplyr)
library(sf)
library(leaflet)
library(bslib)

# -----------------------------
# LOAD YOUR DATA
# -----------------------------
elev_sf <- read_sf("../data/training/03_elevation_m.geojson")

elev_sf$elevation = elev_sf$elevation_m
vars <- read.csv("../data/training/fieldweatherdata_train.csv") 


# -----------------------------
# UI
# -----------------------------
ui <- fluidPage(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  
  titlePanel("US Elevation & Yield Regression Explorer"),
  
  sidebarLayout(
    
    # -----------------------------
    # SIDEBAR 1 — MAP SELECTION
    # -----------------------------
    sidebarPanel(
      h4("Select a point on the map"),
      p("Click anywhere on the US map to view elevation."),
      verbatimTextOutput("clickedCoords")
    ),
    
    mainPanel(
      
      # -----------------------------
      # CARD 1 — MAP + ELEVATION
      # -----------------------------
      card(
        full_screen = TRUE,
        card_header("US Map & Elevation"),
        leafletOutput("usMap", height = 400),
        h5("Elevation at selected point:"),
        verbatimTextOutput("elevOut")
      ),
      
      br(),
      
      # -----------------------------
      # CARD 2 — REGRESSION
      # -----------------------------
      card(
        full_screen = TRUE,
        card_header("Regression with Yield"),
        
        selectInput(
          "var",
          "Select variable:",
          choices = c("ppt", "temp", "yield"),
          selected = "ppt"
        ),
        
        plotOutput("regPlot"),
        verbatimTextOutput("regSummary")
      )
    )
  )
)

# -----------------------------
# SERVER
# -----------------------------
server <- function(input, output, session) {
  
  # -----------------------------
  # LEAFLET MAP WITH ELEVATION POINTS
  # -----------------------------
  output$usMap <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      setView(lng = -98.5, lat = 39.8, zoom = 4) %>%
      
      # Add elevation points
      addCircleMarkers(
        data = elev_sf,
        radius = 4,
        color = "red",
        fillOpacity = 0.7,
        popup = ~paste0(
          "<b>Elevation:</b> ", elevation, " m<br>",
          "<b>Lon:</b> ", st_coordinates(elev_sf)[,1], "<br>",
          "<b>Lat:</b> ", st_coordinates(elev_sf)[,2]
        )
      )
  })
  
  # Store clicked location
  click_reactive <- reactiveVal(NULL)
  
  observeEvent(input$usMap_click, {
    click_reactive(input$usMap_click)
  })
  
  output$clickedCoords <- renderPrint({
    req(click_reactive())
    click_reactive()
  })
  
  # -----------------------------
  # FIND NEAREST ELEVATION POINT
  # -----------------------------
  output$elevOut <- renderPrint({
    req(click_reactive())
    
    click <- click_reactive()
    click_sf <- st_as_sf(
      data.frame(lon = click$lng, lat = click$lat),
      coords = c("lon", "lat"), crs = 4326
    )
    
    # nearest elevation point
    idx <- st_nearest_feature(click_sf, elev_sf)
    elev$elevation[idx]
  })
  
  # -----------------------------
  # REGRESSION WITH YIELD
  # -----------------------------
  output$regPlot <- renderPlot({
    req(input$var)
    
    ggplot(vars, aes_string(x = input$var, y = "yield")) +
      geom_point(color = "steelblue") +
      geom_smooth(method = "lm", se = TRUE, color = "darkred") +
      labs(
        title = paste("Regression of", input$var, "vs Yield"),
        x = input$var,
        y = "Yield"
      ) +
      theme_minimal()
  })
  
  output$regSummary <- renderPrint({
    req(input$var)
    model <- lm(as.formula(paste("yield ~", input$var)), data = vars)
    summary(model)
  })
}

shinyApp(ui, server)
