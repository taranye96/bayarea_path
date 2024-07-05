#Grace Parker
#March 8, 2023
#Want to create a simulated example of fitting a mixed-effects path model

libs <- c("ggplot2","lme4")
lapply(libs, require, character.only = T)

# Read in residual file
real.data <- read.csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/meml_test.csv',sep=",")
real.data$GMs_res = log(real.data$GMs_total / real.data$GMs_mean)
# Fit mixed-effect model to data ------------------------------------------

#Define column with logR to fit to
real.data$logR <- log(real.data$Rs)

#Leading 1 = fixed effect intercept, (1|EQID) = event term, (1|StationId) is site term, 
#1 + logR|site class fits an intercept (1) and slope (logR) for the basin and non-basin sites (based on SiteClass)
ctrl = lmerControl(optimizer = "Nelder_Mead")
#fit <- lmer(GMs_total ~  1 + (1|EQID) + (1|StationID),
#                data = real.data,
#                control = ctrl)
fit <- lmer(GMs_res ~ 1 + (1|EQID) + (1|StationID),
            data = real.data, control = ctrl)
summary(fit)
FE <- fixef(fit)
RanR <- as.data.frame(ranef(fit, condVar = T))

#Add model line to plot
Rrup <- seq(1,100,by = 1)
R <- sqrt(h^2 + Rrup^2)
logR <- log(R)

basin.line <- FE + RanR$condval[RanR$term == "logR" & RanR$grp =="Basin"]*logR
nonbasin.line <- FE + RanR$condval[RanR$term == "logR" & RanR$grp =="Non_Basin"]*logR

##############################################################
#Plot for sanity check
data.plot <- ggplot() + 
  geom_point(aes(x= Rrups, y = GMs_total),
             data = real.data) +
  scale_x_continuous(trans = "log") +
  ylab("Log of GM") +
  xlab("Rupture Distance (km)") + 
  theme(axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16),
        axis.text.x = element_text(size = 16, color = "black"),
        axis.text.y = element_text(size = 16,color = "black"),
        axis.ticks.length=unit(.25, "cm"),
        panel.border = element_rect(colour = "black", fill=NA, size=0.5))
data.plot

data.plot +
  geom_line(aes(x = Rrup, y = basin.line)) +
  geom_line(aes(x = Rrup, y = nonbasin.line))


plot(real.data$Rrups, real.data$GMs_total*9.8,
     log = "xy",            # Set logarithmic scale for both x and y axes
     main = "Log-Log Scatter Plot",  # Title of the plot
     xlab = "Rrup (km)",    # Label for the x-axis
     ylab = "PGA (m/s**2)",    # Label for the y-axis
     col = "blue",          # Color of the points
     pch = 16,              # Type of points (16 is solid circle)
     xlim = c(1, 500),      # X-axis limits
)