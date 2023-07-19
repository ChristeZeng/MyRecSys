package com.myrecsys.online;

import com.myrecsys.online.datamanager.DataManager;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.util.resource.Resource;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

public class RecSysServer {
    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception {
        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        // set ip address and port number
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        // get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null) {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        // set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$", "/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        // set DataManager
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv", webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        // create server context
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });
        context.getMimeTypes().addMimeMapping("txt", "text/plain;charset=utf-8");

    }
}