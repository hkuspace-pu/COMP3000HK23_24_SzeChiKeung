
using WebSniffer.Discord;

namespace WebSniffer
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            //test a
            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseAuthorization();

            app.MapControllers();

            DCManager.InitToken();
            Task.Run(() => { DCManager.MainAsync().GetAwaiter().GetResult(); });
            Task.Run(() => { app.Run(); });

            while (true)
            {
                var cmd = Console.ReadLine().ToLower();
                if (cmd == "exit") break;
                else
                {
                    switch (cmd)
                    {

                    }
                }
            }
        }
    }
}