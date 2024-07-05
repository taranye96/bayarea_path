#Grace Parker
#March 8, 2023
#Want to create a simulated example of fitting a mixed-effects path model

libs <- c("ggplot2","lme4")
lapply(libs, require, character.only = T)

# Simulate dataset with two site classes and 40 stations ------------------
nquakes <- 20
nstations <- 40

#Set up dataframe
simulated.data <- data.frame(EQID = rep(seq(1,nquakes,by = 1),nstations))
simulated.data$EQID <- simulated.data$EQID[order(simulated.data$EQID)] 
simulated.data$StationID <- rep(seq(1,nstations,by = 1),nquakes)
simulated.data <- simulated.data[order(simulated.data$StationID),] 

#Assign half the stations as basin and half as non-basin
simulated.data$SiteClass[1:400] <- "Basin"
simulated.data$SiteClass[401:800] <- "Non_Basin"

#Define distances for each record using U(1,100) km
simulated.data$Rrups <- runif(nquakes*nstations, min = 1, max = 100)

#Define median ground motion, with arbitrary intercept and 1/R geometrical spreading for basin sites, 1/R^1.5 for non-basin sites, h = 3km
h <- 3
simulated.data$Rs <- sqrt(simulated.data$Rrups^2 + h^2)
simulated.data$GMs_mean[1:400] <- 0.4 -1*log(simulated.data$Rs[1:400])
simulated.data$GMs_mean[401:800] <- 1.1 -1.5*log(simulated.data$Rs[401:800])

#Define event terms and site terms and epsilon
for(i in 1:nquakes){
  simulated.data$ET[simulated.data$EQID == i] <- rnorm(1,mean = 0, sd = 0.2)
}
for(j in 1:nstations){
  simulated.data$ST[simulated.data$StationID == j] <- rnorm(1,mean = 0, sd = 0.4)
}
simulated.data$epsilon <- rnorm(length(simulated.data$EQID), mean = 0, sd = 0.2)

#Now add it up
simulated.data$GMs_total <- simulated.data$GMs_mean + simulated.data$ET + simulated.data$ST + simulated.data$epsilon

#Plot for sanity check
data.plot <- ggplot() + 
  geom_point(aes(x= Rrups, y = GMs_total, color = SiteClass),
             data = simulated.data) +
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

# Fit mixed-effect model to data ------------------------------------------

#Define column with logR to fit to
simulated.data$logR <- log(simulated.data$Rs)

#Leading 1 = fixed effect intercept, (1|EQID) = event term, (1|StationId) is site term, 
#1 + logR|site class fits an intercept (1) and slope (logR) for the basin and non-basin sites (based on SiteClass)
ctrl = lmerControl(optimizer = "Nelder_Mead")
fit <- lmer(GMs_total ~  1 + (1|EQID) + (1|StationID) + (0+ logR|SiteClass),
                data = simulated.data,
                control = ctrl)
summary(fit)
FE <- fixef(fit)
RanR <- as.data.frame(ranef(fit, condVar = T))

#Add model line to plot
Rrup <- seq(1,100,by = 1)
R <- sqrt(h^2 + Rrup^2)
logR <- log(R)

basin.line <- FE + RanR$condval[RanR$term == "logR" & RanR$grp =="Basin"]*logR
nonbasin.line <- FE + RanR$condval[RanR$term == "logR" & RanR$grp =="Non_Basin"]*logR

data.plot +
  geom_line(aes(x = Rrup, y = basin.line)) +
  geom_line(aes(x = Rrup, y = nonbasin.line))
